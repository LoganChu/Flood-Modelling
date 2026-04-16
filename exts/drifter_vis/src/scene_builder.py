"""
scene_builder.py — Build the USD stage for the River Drifter Visualisation.

USD stage structure:
    /World
      /Cesium
        CesiumGeoreference
        CesiumWorldTerrain   (or absent if Cesium unavailable)
        BingMapsSatellite    (or absent if Cesium unavailable)
      /River
        WaterPlane           (fallback flat water surface)
        Trajectory           (BasisCurves — speed-gradient colour)
        TrajectoryPhysics    (BasisCurves — discrepancy colour, optional)
      /Drifter
        DrifterXform         (animated Xform)
          DrifterMesh        (Sphere primitive)
          ChaseCamera
          OnboardCamera
      /Cameras
        OverviewCamera
      /Lighting
        SkyDome
        SunLight

All USD operations are guarded by try/except so the module can be imported
outside Isaac Sim (e.g. during unit testing) without crashing.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from .utils import (
    WORLD_PATH, CESIUM_PATH, RIVER_PATH, DRIFTER_PATH, DRIFTER_XFORM_PATH,
    CAMERAS_PATH, LIGHTING_PATH, TRAJECTORY_PATH, TRAJECTORY_PHYSICS_PATH,
    TRAJECTORY_EKF_PATH, WATER_PLANE_PATH, speed_to_rgb, discrepancy_to_rgb,
    check_dependencies,
)

log = logging.getLogger(__name__)

# Lazy-import Omniverse modules so this file can be imported without Kit
try:
    import omni.usd
    import omni.kit.commands
    from pxr import (
        Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade,
        Vt, Kind,
    )
    _USD_AVAILABLE = True
except ImportError:
    _USD_AVAILABLE = False
    log.debug("USD/Omniverse not available — SceneBuilder in stub mode")


class SceneBuilder:
    """
    Construct and populate the USD stage for the river drifter visualisation.

    Parameters
    ----------
    origin_lat, origin_lon, origin_alt : WGS84 georeference origin
    """

    # Exposed prim paths for other modules to reference
    DRIFTER_XFORM_PATH = DRIFTER_XFORM_PATH
    DRIFTER_PATH = DRIFTER_PATH
    CAMERAS_PATH = CAMERAS_PATH
    TRAJECTORY_PATH = TRAJECTORY_PATH
    TRAJECTORY_PHYSICS_PATH = TRAJECTORY_PHYSICS_PATH

    # Drifter geometry (sphere)
    BUOY_RADIUS_M: float = 0.08

    def __init__(
        self,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        origin_alt: float = 0.0,
    ) -> None:
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._origin_alt = origin_alt
        log.info("SceneBuilder.__init__: initialized with georeference origin - lat=%.5f, lon=%.5f, alt=%.1f",
                origin_lat, origin_lon, origin_alt)
        self._deps = check_dependencies()
        self._stage: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        speeds: np.ndarray,
        physics_east: Optional[np.ndarray] = None,
        physics_north: Optional[np.ndarray] = None,
        discrepancy: Optional[np.ndarray] = None,
        ekf_east: Optional[np.ndarray] = None,
        ekf_north: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build the full USD scene.

        Parameters
        ----------
        east_arr, north_arr : ENU coordinates (metres), arrays of shape (N,)
        speeds          : speed_ms values, array of shape (N,), used for trajectory colour
        physics_east/north : optional simulated ENU coordinates for physics trajectory
        discrepancy     : optional metres-error array for physics trajectory colour
        ekf_east/north  : optional EKF-filtered ENU positions for the EKF overlay

        Note: All trajectories are initially flat at Y=0. Terrain height draping
        is applied asynchronously via TerrainDraper after tiles load.
        """
        if not _USD_AVAILABLE:
            log.warning("USD not available — SceneBuilder.build() is a no-op")
            return

        self._stage = omni.usd.get_context().get_stage()
        stage = self._stage

        # Set stage metadata
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        log.info("Building USD scene…")
        self._build_world_xform(stage)
        self._build_lighting(stage)
        self._build_terrain(stage)
        self._build_trajectory(stage, east_arr, north_arr, speeds)
        self._build_drifter(stage)
        self._build_cameras(stage, east_arr, north_arr)

        if physics_east is not None and discrepancy is not None:
            self._build_physics_trajectory(
                stage, physics_east, physics_north or np.zeros_like(physics_east),
                discrepancy
            )

        if ekf_east is not None and ekf_north is not None:
            self._build_ekf_trajectory(stage, ekf_east, ekf_north)

        # Enable physics colliders on Cesium terrain (if present) for TerrainDraper raycasting
        self._enable_terrain_colliders(stage)

        log.info("USD scene build complete.")

    # ------------------------------------------------------------------
    # Stage sections
    # ------------------------------------------------------------------

    def _build_world_xform(self, stage) -> None:
        world = stage.DefinePrim(WORLD_PATH, "Xform")
        UsdGeom.Xform(world)

    def _build_lighting(self, stage) -> None:
        lighting_path = LIGHTING_PATH

        # Check if lighting already exists (skip if rebuilding scene)
        existing_dome = stage.GetPrimAtPath(f"{lighting_path}/SkyDome")
        if existing_dome.IsValid():
            log.debug("Lighting already exists, skipping rebuild")
            return

        # Sky dome (environment light)
        dome_path = f"{lighting_path}/SkyDome"
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(1000.0)
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))  # light blue sky
        dome.CreateTextureFileAttr("")                    # placeholder; swap with HDR

        # Directional sun light — Feb 16, 36°N, midday azimuth ≈ 180°S, elevation ≈ 38°
        sun_path = f"{lighting_path}/SunLight"
        sun = UsdLux.DistantLight.Define(stage, sun_path)
        sun.CreateIntensityAttr(5000.0)
        sun.CreateAngleAttr(0.53)   # solar disk angular diameter (degrees)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))

        # Rotate sun to elevation 38° from south in Y-up stage
        # azimuth 180° (south) + elevation 38°: rotate Y-up stage xform
        xform = UsdGeom.Xformable(sun.GetPrim())
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 180.0, 0.0))

        log.debug("Lighting prims created")

    def _build_terrain(self, stage) -> None:
        if self._deps.get("cesium"):
            self._build_cesium_terrain(stage)
        else:
            log.info("Cesium not available — using flat water plane fallback")
            self._build_water_plane(stage)

    def _build_cesium_terrain(self, stage) -> None:
        """
        Check if Cesium prims are already set up in the scene.

        If Cesium World Terrain exists, validate/set its georeference and reparent
        it under /World so it shares the same coordinate hierarchy as the trajectory.
        Otherwise, fall back to water plane and log instructions for manual setup.
        """
        from .utils import CESIUM_TERRAIN_PATH, WORLD_PATH

        log.info("_build_cesium_terrain: START - checking for Cesium terrain at %s", CESIUM_TERRAIN_PATH)

        # Check if Cesium terrain was manually added to the scene
        cesium_prim = stage.GetPrimAtPath(CESIUM_TERRAIN_PATH)
        if cesium_prim.IsValid():
            log.info("_build_cesium_terrain: Cesium World Terrain FOUND at %s (type=%s)",
                    CESIUM_TERRAIN_PATH, cesium_prim.GetTypeName())

            # Reparent Cesium terrain under /World if it's at root level
            # This ensures raycasts work correctly (both terrain and trajectory in same coord space)
            parent = cesium_prim.GetParent()
            parent_path = str(parent.GetPath()) if parent else "/"
            if parent_path != WORLD_PATH:
                log.info("_build_cesium_terrain: Cesium terrain is at root level (%s), "
                        "reparenting under %s for raycast coordinate alignment",
                        parent_path, WORLD_PATH)
                world_prim = stage.GetPrimAtPath(WORLD_PATH)
                if world_prim.IsValid():
                    try:
                        cesium_name = cesium_prim.GetName()
                        old_path = cesium_prim.GetPath()
                        new_path = Sdf.Path(f"{WORLD_PATH}/{cesium_name}")

                        # Use Sdf layer operations to move the prim spec
                        root_layer = stage.GetRootLayer()
                        old_spec = root_layer.GetPrimAtPath(old_path)
                        if old_spec:
                            # Move spec in the layer hierarchy
                            Sdf.BatchNamespaceEdit([
                                Sdf.NamespaceEdit.Reparent(old_path, new_path.GetParentPath(), new_path.GetName())
                            ]).Apply(root_layer)
                            log.info("_build_cesium_terrain: reparented from %s to %s", old_path, new_path)

                            # Fetch prim at new location (don't reload stage to avoid invalidating other prims)
                            cesium_prim = stage.GetPrimAtPath(str(new_path))
                            if cesium_prim.IsValid():
                                log.info("_build_cesium_terrain: prim now accessible at new location")
                        else:
                            log.info("_build_cesium_terrain: prim spec not found at %s", old_path)
                    except Exception as exc:
                        log.info("_build_cesium_terrain: reparenting failed: %s (continuing anyway)", exc)

            # Try to set/validate Cesium Georeference
            self._sync_cesium_georeference(stage)
            log.info("_build_cesium_terrain: COMPLETE - using Cesium terrain")
            return  # Don't build water plane; use existing Cesium

        # Cesium not set up yet; log instructions and fall back
        log.info(
            "_build_cesium_terrain: Cesium terrain NOT FOUND at %s, falling back to water plane. "
            "Expected CSV origin: lat=%.5f, lon=%.5f, height=%.1f",
            CESIUM_TERRAIN_PATH,
            self._origin_lat, self._origin_lon, self._origin_alt
        )
        self._build_water_plane(stage)

    def _sync_cesium_georeference(self, stage) -> None:
        """
        Set Cesium Georeference to match the CSV data's origin (lat/lon/alt).

        This ensures the drifter trajectory aligns with Cesium terrain tiles.
        """
        log.info("_sync_cesium_georeference: START")
        try:
            cesium_georef_path = "/CesiumGeoreference"
            log.info("_sync_cesium_georeference: looking for prim at %s", cesium_georef_path)

            georef_prim = stage.GetPrimAtPath(cesium_georef_path)
            if not georef_prim.IsValid():
                log.info("_sync_cesium_georeference: prim NOT FOUND/INVALID at %s", cesium_georef_path)
                return

            log.info("_sync_cesium_georeference: prim found and valid, type=%s", georef_prim.GetTypeName())

            # Set latitude, longitude, height to match CSV origin
            lat_attr = georef_prim.GetAttribute("cesium:georeferenceOrigin:latitude")
            lon_attr = georef_prim.GetAttribute("cesium:georeferenceOrigin:longitude")
            height_attr = georef_prim.GetAttribute("cesium:georeferenceOrigin:height")

            log.info("_sync_cesium_georeference: attribute lookup results - lat=%s, lon=%s, height=%s",
                    "found" if lat_attr else "NOT FOUND",
                    "found" if lon_attr else "NOT FOUND",
                    "found" if height_attr else "NOT FOUND")

            success_count = 0

            if lat_attr:
                try:
                    old_val = lat_attr.Get()
                    log.info("_sync_cesium_georeference: setting cesium:latitude from %.5f to %.5f", old_val, self._origin_lat)
                    lat_attr.Set(self._origin_lat)
                    new_val = lat_attr.Get()
                    log.info("_sync_cesium_georeference: cesium:latitude SET SUCCESS - verified new value: %.5f", new_val)
                    success_count += 1
                except Exception as exc:
                    log.info("_sync_cesium_georeference: FAILED to set cesium:latitude: %s", exc)
            else:
                log.info("_sync_cesium_georeference: cesium:latitude attribute not found")

            if lon_attr:
                try:
                    old_val = lon_attr.Get()
                    log.info("_sync_cesium_georeference: setting cesium:longitude from %.5f to %.5f", old_val, self._origin_lon)
                    lon_attr.Set(self._origin_lon)
                    new_val = lon_attr.Get()
                    log.info("_sync_cesium_georeference: cesium:longitude SET SUCCESS - verified new value: %.5f", new_val)
                    success_count += 1
                except Exception as exc:
                    log.info("_sync_cesium_georeference: FAILED to set cesium:longitude: %s", exc)
            else:
                log.info("_sync_cesium_georeference: cesium:longitude attribute not found")

            if height_attr:
                try:
                    old_val = height_attr.Get()
                    log.info("_sync_cesium_georeference: setting cesium:height from %.1f to %.1f", old_val, self._origin_alt)
                    height_attr.Set(self._origin_alt)
                    new_val = height_attr.Get()
                    log.info("_sync_cesium_georeference: cesium:height SET SUCCESS - verified new value: %.1f", new_val)
                    success_count += 1
                except Exception as exc:
                    log.info("_sync_cesium_georeference: FAILED to set cesium:height: %s", exc)
            else:
                log.info("_sync_cesium_georeference: cesium:height attribute not found")

            log.info(
                "_sync_cesium_georeference: COMPLETE - successfully set %d/3 attributes (lat=%.5f, lon=%.5f, height=%.1f)",
                success_count,
                self._origin_lat, self._origin_lon, self._origin_alt
            )
        except Exception as exc:
            log.info("_sync_cesium_georeference: FAILED with exception: %s", exc)

    def _build_water_plane(self, stage) -> None:
        """Flat 500×500 m water plane as Cesium fallback."""
        plane_path = WATER_PLANE_PATH

        mesh = UsdGeom.Mesh.Define(stage, plane_path)
        half = 250.0
        points = [
            Gf.Vec3f(-half, 0.0, -half),
            Gf.Vec3f( half, 0.0, -half),
            Gf.Vec3f( half, 0.0,  half),
            Gf.Vec3f(-half, 0.0,  half),
        ]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # Water material — translucent blue PBR
        mat_path = f"{WATER_PLANE_PATH}/WaterMaterial"
        mat, shader = self._create_preview_surface_material(stage, mat_path)
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.05, 0.25, 0.45))
        shader.GetInput("roughness").Set(0.05)
        shader.GetInput("metallic").Set(0.0)
        shader.GetInput("opacity").Set(0.75)
        shader.GetInput("ior").Set(1.33)
        UsdShade.MaterialBindingAPI(mesh).Bind(mat)

        log.debug("Water plane created at %s", plane_path)

    def _build_trajectory(
        self,
        stage,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        speeds: np.ndarray,
    ) -> None:
        """
        Create a BasisCurves prim coloured by speed (viridis gradient).

        The full recorded path is baked as a single linear curves prim with
        per-vertex displayColor — one GPU draw call, no per-frame cost.

        All points are initially flat at Y=0. Terrain height is drape via
        TerrainDraper after tiles load.
        """
        traj = UsdGeom.BasisCurves.Define(stage, TRAJECTORY_PATH)
        traj.GetTypeAttr().Set(UsdGeom.Tokens.linear)

        # Points: (east→X, 0.0→Y, north→Z)
        # Y is flat initially; TerrainDraper will update with terrain heights
        pts = [Gf.Vec3f(float(e), 0.0, float(n))
               for e, n in zip(east_arr, north_arr)]
        traj.GetPointsAttr().Set(Vt.Vec3fArray(pts))
        traj.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))

        # Vertex colours — speed gradient
        max_spd = float(np.max(speeds)) if len(speeds) > 0 else 1.0
        colours = [Gf.Vec3f(*speed_to_rgb(float(s), max_spd)) for s in speeds]
        traj.CreateDisplayColorAttr(Vt.Vec3fArray(colours))
        traj.CreateWidthsAttr(Vt.FloatArray([0.05] * len(pts)))

        log.debug("Trajectory curve created with %d points", len(pts))

    def _build_physics_trajectory(
        self,
        stage,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        discrepancy: np.ndarray,
    ) -> None:
        """Optional physics comparison trajectory, coloured by discrepancy.

        Positioned 0.3 m above the GPS trajectory for visual separation.
        Y values are initially at 0.3; TerrainDraper will update to terrain + 0.3.
        """
        traj = UsdGeom.BasisCurves.Define(stage, TRAJECTORY_PHYSICS_PATH)
        traj.GetTypeAttr().Set(UsdGeom.Tokens.linear)

        pts = [Gf.Vec3f(float(e), 0.3, float(n))
               for e, n in zip(east_arr, north_arr)]  # offset +0.3m Y
        traj.GetPointsAttr().Set(Vt.Vec3fArray(pts))
        traj.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))

        colours = [Gf.Vec3f(*discrepancy_to_rgb(float(d))) for d in discrepancy]
        traj.CreateDisplayColorAttr(Vt.Vec3fArray(colours))
        traj.CreateWidthsAttr(Vt.FloatArray([0.05] * len(pts)))

        log.debug("Physics trajectory created")

    def _build_ekf_trajectory(
        self,
        stage,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
    ) -> None:
        """EKF-filtered trajectory overlay, constant green, offset +0.6 m Y.

        Positioned 0.6 m above the GPS trajectory for visual separation.
        Y values are initially at 0.6; TerrainDraper will update to terrain + 0.6.
        """
        traj = UsdGeom.BasisCurves.Define(stage, TRAJECTORY_EKF_PATH)
        traj.GetTypeAttr().Set(UsdGeom.Tokens.linear)

        pts = [Gf.Vec3f(float(e), 0.6, float(n))
               for e, n in zip(east_arr, north_arr)]
        traj.GetPointsAttr().Set(Vt.Vec3fArray(pts))
        traj.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))

        # Constant green — visually distinct from GPS (viridis) and physics (orange)
        colour = Gf.Vec3f(0.1, 0.8, 0.3)
        traj.CreateDisplayColorAttr(Vt.Vec3fArray([colour] * len(pts)))
        traj.CreateWidthsAttr(Vt.FloatArray([0.05] * len(pts)))

        log.debug("EKF trajectory created")

    def _build_drifter(self, stage) -> None:
        """Create the drifter Xform hierarchy with sphere mesh (cameras now separate)."""
        # Root Xform (animated by Animator)
        xform_prim = stage.DefinePrim(DRIFTER_XFORM_PATH, "Xform")
        UsdGeom.Xform(xform_prim)

        # Sphere mesh
        mesh_path = f"{DRIFTER_XFORM_PATH}/DrifterMesh"
        sphere = UsdGeom.Sphere.Define(stage, mesh_path)
        sphere.GetRadiusAttr().Set(self.BUOY_RADIUS_M)

        # Orange buoy material
        mat_path = f"{mesh_path}/BuoyMaterial"
        mat, shader = self._create_preview_surface_material(stage, mat_path)
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.55, 0.1))
        shader.GetInput("roughness").Set(0.6)
        shader.GetInput("metallic").Set(0.05)
        UsdShade.MaterialBindingAPI(sphere).Bind(mat)

        log.debug("Drifter prim hierarchy created")

    def _build_cameras(
        self,
        stage,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
    ) -> None:
        """Create orbit and follow cameras above the path centroid.

        Chase and Onboard cameras follow the drifter's path without rotating,
        maintaining a fixed horizontal orientation at all times.
        """
        # Remove existing camera prims if rebuilding scene
        for cam_path in [f"{CAMERAS_PATH}/OverviewCamera", f"{CAMERAS_PATH}/ChaseCamera", f"{CAMERAS_PATH}/OnboardCamera"]:
            existing = stage.GetPrimAtPath(cam_path)
            if existing.IsValid():
                stage.RemovePrim(cam_path)

        # Overview camera (fixed, at trajectory start) — positioned high to load Cesium tiles
        east_min = float(east_arr.min())
        east_max = float(east_arr.max())
        north_min = float(north_arr.min())
        north_max = float(north_arr.max())

        # Position camera over the START of the trajectory (first data point)
        cx = float(east_arr[0])
        cz = float(north_arr[0])
        cy = 100.0

        extent = max(east_max - east_min, north_max - north_min)
        # Height positioned to view trajectory start and extend along the path
        orbit_h = extent * 1.5

        log.info("Overview camera: trajectory bounds east=[%.2f, %.2f], north=[%.2f, %.2f], extent=%.2f",
                east_min, east_max, north_min, north_max, extent)
        log.info("Overview camera: positioned at start (%.2f, %.2f, %.2f) height=%.2f for Cesium tile loading",
                cx, orbit_h, cz, orbit_h)

        overview_path = f"{CAMERAS_PATH}/OverviewCamera"
        cam = UsdGeom.Camera.Define(stage, overview_path)
        # Wider field of view to ensure trajectory from start is visible
        cam.GetFocalLengthAttr().Set(24.0)
        cam.GetHorizontalApertureAttr().Set(36.0)

        cam_xform = UsdGeom.Xformable(cam.GetPrim())
        # Position above START point at high altitude, looking down along the trajectory
        # Use -90° X rotation (tip back) to look down at the ground plane below
        cam_xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy + orbit_h, cz))
        cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

        log.info("Overview camera at (%.1f, %.1f, %.1f) with height=%.1f m", cx, cy + orbit_h, cz, orbit_h)

        # Chase and Onboard cameras: follow drifter without rotating
        self._build_follow_cameras(stage, east_arr, north_arr)

    def _build_follow_cameras(
        self,
        stage,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
    ) -> None:
        """Create Chase and Onboard cameras that follow the drifter without rotating.

        Cameras are positioned at each path point with time samples, maintaining
        a fixed horizontal orientation throughout the animation.

        Chase: 5m behind the drifter, 2m up, looking forward (unrotated).
        Onboard: 0.3m above drifter, first-person view (unrotated).
        """
        n_points = len(east_arr)

        # Compute path heading (direction of travel) at each point
        headings = np.zeros(n_points, dtype=float)
        for i in range(n_points - 1):
            de = float(east_arr[i + 1] - east_arr[i])
            dn = float(north_arr[i + 1] - north_arr[i])
            heading = math.atan2(de, dn)  # radians: 0=north, π/2=east
            headings[i] = heading
        headings[-1] = headings[-2]  # repeat last heading at end

        # Chase camera: 5m behind drifter, 2m up, always horizontal
        chase_path = f"{CAMERAS_PATH}/ChaseCamera"
        chase_cam = UsdGeom.Camera.Define(stage, chase_path)
        chase_cam.GetFocalLengthAttr().Set(24.0)
        chase_cam.GetHorizontalApertureAttr().Set(36.0)

        chase_xform = UsdGeom.Xformable(chase_cam.GetPrim())
        chase_trans_op = chase_xform.AddTranslateOp()
        chase_rot_op = chase_xform.AddRotateXYZOp()

        for i in range(n_points):
            e = float(east_arr[i])
            n = float(north_arr[i])
            heading = float(headings[i])

            # Offset: 5m backward along the heading direction
            chase_e = e - 5.0 * math.sin(heading)
            chase_n = n - 5.0 * math.cos(heading)
            chase_y = 2.0

            time_code = Usd.TimeCode(i)
            chase_trans_op.Set(Gf.Vec3d(chase_e, chase_y, chase_n), time_code)
            chase_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0), time_code)  # Horizontal

        # Onboard camera: 0.3m above drifter, always horizontal
        onboard_path = f"{CAMERAS_PATH}/OnboardCamera"
        onboard_cam = UsdGeom.Camera.Define(stage, onboard_path)
        onboard_cam.GetFocalLengthAttr().Set(18.0)
        onboard_cam.GetHorizontalApertureAttr().Set(36.0)

        onboard_xform = UsdGeom.Xformable(onboard_cam.GetPrim())
        onboard_trans_op = onboard_xform.AddTranslateOp()
        onboard_rot_op = onboard_xform.AddRotateXYZOp()

        for i in range(n_points):
            e = float(east_arr[i])
            n = float(north_arr[i])

            time_code = Usd.TimeCode(i)
            onboard_trans_op.Set(Gf.Vec3d(e, 0.3, n), time_code)
            onboard_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0), time_code)  # Horizontal

        log.debug("Chase and Onboard cameras created with %d time samples", n_points)

    # ------------------------------------------------------------------
    # Terrain colliders (for TerrainDraper raycasting)
    # ------------------------------------------------------------------

    def _enable_terrain_colliders(self, stage) -> bool:
        """
        Apply UsdPhysics.CollisionAPI to the CesiumWorldTerrain prim if it exists.

        Also sets cesium:enableFrustumCulling = false so the tile loader
        fetches the full tileset footprint regardless of viewport frustum.

        Returns True if the prim was found and APIs applied, False otherwise.
        """
        if not _USD_AVAILABLE:
            log.info("_enable_terrain_colliders: USD not available, returning False")
            return False

        from .utils import CESIUM_TERRAIN_PATH
        log.info("_enable_terrain_colliders: looking for prim at %s", CESIUM_TERRAIN_PATH)

        prim = stage.GetPrimAtPath(CESIUM_TERRAIN_PATH)
        if not prim.IsValid():
            log.info("_enable_terrain_colliders: CesiumWorldTerrain prim NOT FOUND/INVALID at %s", CESIUM_TERRAIN_PATH)
            return False

        log.info("_enable_terrain_colliders: found prim at %s, type=%s", CESIUM_TERRAIN_PATH, prim.GetTypeName())

        # Apply collision API
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            log.info("_enable_terrain_colliders: CollisionAPI not present, applying it...")
            UsdPhysics.CollisionAPI.Apply(prim)
            log.info("_enable_terrain_colliders: Applied UsdPhysics.CollisionAPI to %s", CESIUM_TERRAIN_PATH)
        else:
            log.info("_enable_terrain_colliders: CollisionAPI already present on %s", CESIUM_TERRAIN_PATH)

        # Disable frustum culling so tiles load outside the viewport frustum
        frustum_attr = prim.GetAttribute("cesium:enableFrustumCulling")
        if not frustum_attr:
            log.info("_enable_terrain_colliders: creating cesium:enableFrustumCulling attribute...")
            frustum_attr = prim.CreateAttribute(
                "cesium:enableFrustumCulling",
                Sdf.ValueTypeNames.Bool,
                custom=True
            )
        frustum_attr.Set(False)
        log.info("_enable_terrain_colliders: cesium:enableFrustumCulling set to False")

        log.info("_enable_terrain_colliders: SUCCESS - Cesium terrain physics colliders enabled at %s", CESIUM_TERRAIN_PATH)
        return True

    def enable_terrain_colliders(self) -> bool:
        """Public accessor — re-applies collision API after deferred tile load."""
        if not self._stage:
            return False
        return self._enable_terrain_colliders(self._stage)

    # ------------------------------------------------------------------
    # Material helper
    # ------------------------------------------------------------------

    @staticmethod
    def _create_preview_surface_material(
        stage, mat_path: str
    ) -> Tuple[object, object]:
        """
        Create a UsdPreviewSurface material + shader pair.

        Returns (Material, Shader) prims ready for attribute setting.
        """
        mat = UsdShade.Material.Define(stage, mat_path)
        shader_path = f"{mat_path}/PBRShader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f)
        shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity",       Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("ior",           Sdf.ValueTypeNames.Float).Set(1.5)
        mat.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        return mat, shader
