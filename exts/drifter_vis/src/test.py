from pxr import UsdGeom, Gf
import omni.usd

# Get stage
stage = omni.usd.get_context().get_stage()

# Create a simple line (trajectory placeholder)
points = [
    Gf.Vec3f(0, 0, 0),
    Gf.Vec3f(1, 1, 0),
    Gf.Vec3f(2, 1, 0),
]

curve = UsdGeom.BasisCurves.Define(stage, "/World/Trajectory")

curve.CreatePointsAttr(points)
curve.CreateCurveVertexCountsAttr([len(points)])
curve.CreateTypeAttr("linear")
curve.CreateWidthsAttr([0.05])