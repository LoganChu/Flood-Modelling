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
          DrifterMesh        (Cylinder primitive)
          WakeParticles      (OmniParticles emitter, if available)
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
    WATER_PLANE_PATH, speed_to_rgb, discrepancy_to_rgb, check_dependencies,
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

    # Drifter geometry
    BUOY_RADIUS_M: float = 0.15
    BUOY_HEIGHT_M: float = 0.30

    def __init__(
        self,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        origin_alt: float = 0.0,
    ) -> None:
        self._origin_lat = origin_lat
        self._origin_lon = origin_lon
        self._origin_alt = origin_alt
        self._deps = check_dependencies()
        self._stage: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        up_arr: np.ndarray,
        speeds: np.ndarray,
        physics_east: Optional[np.ndarray] = None,
        physics_north: Optional[np.ndarray] = None,
        discrepancy: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build the full USD scene.

        Parameters
        ----------
        east_arr, north_arr, up_arr : ENU coordinates (metres), arrays of shape (N,)
        speeds          : speed_ms values, array of shape (N,), used for trajectory colour
        physics_east/north : optional simulated ENU coordinates for physics trajectory
        discrepancy     : optional metres-error array for physics trajectory colour
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
        self._build_trajectory(stage, east_arr, north_arr, up_arr, speeds)
        self._build_drifter(stage)
        self._build_cameras(stage, east_arr, north_arr, up_arr)

        if physics_east is not None and discrepancy is not None:
            self._build_physics_trajectory(
                stage, physics_east, physics_north or np.zeros_like(physics_east),
                up_arr, discrepancy
            )

        log.info("USD scene build complete.")

    # ------------------------------------------------------------------
    # Stage sections
    # ------------------------------------------------------------------

    def _build_world_xform(self, stage) -> None:
        world = stage.DefinePrim(WORLD_PATH, "Xform")
        UsdGeom.Xform(world)

    def _build_lighting(self, stage) -> None:
        lighting_path = LIGHTING_PATH

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
        Add Cesium georeferenced terrain and satellite imagery.

        Requires the cesium.omniverse Kit extension to be enabled.
        Ion assets used:
          1   — Cesium World Terrain
          2   — Bing Maps Aerial imagery
        """
        try:
            import cesium.omniverse  # noqa: F401
            from cesium.omniverse.utils.usd_utils import add_cesium_georeference

            # Georeference prim
            geo_path = f"{CESIUM_PATH}/CesiumGeoreference"
            georef = stage.DefinePrim(geo_path, "CesiumGeoreference")
            georef.GetAttribute("cesium:georeferenceOrigin:latitude").Set(self._origin_lat)
            georef.GetAttribute("cesium:georeferenceOrigin:longitude").Set(self._origin_lon)
            georef.GetAttribute("cesium:georeferenceOrigin:height").Set(self._origin_alt)

            # Terrain tileset
            terrain_path = f"{CESIUM_PATH}/CesiumWorldTerrain"
            terrain = stage.DefinePrim(terrain_path, "CesiumTileset")
            terrain.GetAttribute("cesium:ionAssetId").Set(1)
            terrain.GetAttribute("cesium:maximumScreenSpaceError").Set(8.0)

            # Imagery overlay
            imagery_path = f"{CESIUM_PATH}/BingMapsSatellite"
            imagery = stage.DefinePrim(imagery_path, "CesiumIonRasterOverlay")
            imagery.GetAttribute("cesium:ionAssetId").Set(2)

            log.info("Cesium terrain configured (origin %.5f, %.5f)", self._origin_lat, self._origin_lon)
        except Exception as exc:
            log.warning("Cesium setup failed (%s) — falling back to water plane", exc)
            self._build_water_plane(stage)

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
        up_arr: np.ndarray,
        speeds: np.ndarray,
    ) -> None:
        """
        Create a BasisCurves prim coloured by speed (viridis gradient).

        The full recorded path is baked as a single linear curves prim with
        per-vertex displayColor — one GPU draw call, no per-frame cost.
        """
        traj = UsdGeom.BasisCurves.Define(stage, TRAJECTORY_PATH)
        traj.GetTypeAttr().Set(UsdGeom.Tokens.linear)

        # Points: (east→X, up→Y, north→Z)
        pts = [Gf.Vec3f(float(e), float(u), float(n))
               for e, u, n in zip(east_arr, up_arr, north_arr)]
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
        up_arr: np.ndarray,
        discrepancy: np.ndarray,
    ) -> None:
        """Optional physics comparison trajectory, coloured by discrepancy."""
        traj = UsdGeom.BasisCurves.Define(stage, TRAJECTORY_PHYSICS_PATH)
        traj.GetTypeAttr().Set(UsdGeom.Tokens.linear)

        pts = [Gf.Vec3f(float(e), float(u) + 0.3, float(n))
               for e, u, n in zip(east_arr, up_arr, north_arr)]  # offset +0.3m Y
        traj.GetPointsAttr().Set(Vt.Vec3fArray(pts))
        traj.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))

        colours = [Gf.Vec3f(*discrepancy_to_rgb(float(d))) for d in discrepancy]
        traj.CreateDisplayColorAttr(Vt.Vec3fArray(colours))
        traj.CreateWidthsAttr(Vt.FloatArray([0.05] * len(pts)))

        log.debug("Physics trajectory created")

    def _build_drifter(self, stage) -> None:
        """Create the drifter Xform hierarchy with cylinder mesh and cameras."""
        # Root Xform (animated by Animator)
        xform_prim = stage.DefinePrim(DRIFTER_XFORM_PATH, "Xform")
        UsdGeom.Xform(xform_prim)

        # Cylinder mesh
        mesh_path = f"{DRIFTER_XFORM_PATH}/DrifterMesh"
        cyl = UsdGeom.Cylinder.Define(stage, mesh_path)
        cyl.GetRadiusAttr().Set(self.BUOY_RADIUS_M)
        cyl.GetHeightAttr().Set(self.BUOY_HEIGHT_M)
        cyl.GetAxisAttr().Set(UsdGeom.Tokens.y)

        # Orange buoy material
        mat_path = f"{mesh_path}/BuoyMaterial"
        mat, shader = self._create_preview_surface_material(stage, mat_path)
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.55, 0.1))
        shader.GetInput("roughness").Set(0.6)
        shader.GetInput("metallic").Set(0.05)
        UsdShade.MaterialBindingAPI(cyl).Bind(mat)

        # Chase camera (child of drifter — inherits animated transform)
        chase_path = f"{DRIFTER_XFORM_PATH}/ChaseCamera"
        chase_cam = UsdGeom.Camera.Define(stage, chase_path)
        chase_xform = UsdGeom.Xformable(chase_cam.GetPrim())
        # Offset: 5 m behind (+Z in local drifter space), 2 m up
        chase_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 5.0))
        chase_xform.AddRotateXYZOp().Set(Gf.Vec3f(-15.0, 0.0, 0.0))
        chase_cam.GetFocalLengthAttr().Set(24.0)
        chase_cam.GetHorizontalApertureAttr().Set(36.0)

        # Onboard POV camera
        onboard_path = f"{DRIFTER_XFORM_PATH}/OnboardCamera"
        onboard_cam = UsdGeom.Camera.Define(stage, onboard_path)
        onboard_xform = UsdGeom.Xformable(onboard_cam.GetPrim())
        onboard_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.3, 0.0))
        onboard_xform.AddRotateXYZOp().Set(Gf.Vec3f(-5.0, 0.0, 0.0))
        onboard_cam.GetFocalLengthAttr().Set(18.0)
        onboard_cam.GetHorizontalApertureAttr().Set(36.0)

        log.debug("Drifter prim hierarchy created")

    def _build_cameras(
        self,
        stage,
        east_arr: np.ndarray,
        north_arr: np.ndarray,
        up_arr: np.ndarray,
    ) -> None:
        """Create the orbital overview camera above the path centroid."""
        cx = float(east_arr.mean())
        cy = float(up_arr.mean())
        cz = float(north_arr.mean())
        extent = max(
            float(east_arr.max() - east_arr.min()),
            float(north_arr.max() - north_arr.min()),
        )
        orbit_h = extent * 0.7

        overview_path = f"{CAMERAS_PATH}/OverviewCamera"
        cam = UsdGeom.Camera.Define(stage, overview_path)
        cam.GetFocalLengthAttr().Set(35.0)
        cam.GetHorizontalApertureAttr().Set(36.0)

        cam_xform = UsdGeom.Xformable(cam.GetPrim())
        # Position above centroid, looking down
        cam_xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy + orbit_h, cz))
        cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

        log.debug("Overview camera at (%.1f, %.1f, %.1f) h=%.1f", cx, cy, cz, orbit_h)

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
