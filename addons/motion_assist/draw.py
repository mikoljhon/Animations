# Motion Assist: viewport overlay. The centre of mass (ball and plumb line), its path coloured by what physics says
# about each frame, the support polygon, the contact points and the zero moment point (the balance point).

import bpy
import blf
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import analysis
from . import body as bodymod

STATUS_COL = {
    analysis.FLIGHT: (1.0, 0.55, 0.1, 1.0),
    analysis.OK: (0.25, 0.9, 0.35, 1.0),
    analysis.OFF_BALANCE: (1.0, 0.2, 0.2, 1.0),
    analysis.SLIP: (1.0, 0.85, 0.1, 1.0),
    analysis.PULL: (0.75, 0.35, 1.0, 1.0),
}
GREY = (0.6, 0.6, 0.6, 0.8)
_handles = []
_shaders = {}
_errors = [0]


def _shader(name):
    if name not in _shaders:
        _shaders[name] = gpu.shader.from_builtin(name)
    return _shaders[name]


def _lines(coords, colors, width=2.0, mode='LINE_STRIP'):
    if len(coords) < 2:
        return
    sh = _shader('POLYLINE_FLAT_COLOR')
    sh.bind()
    vp = gpu.state.viewport_get()
    sh.uniform_float("viewportSize", (vp[2], vp[3]))
    sh.uniform_float("lineWidth", width)
    batch_for_shader(sh, mode, {"pos": coords, "color": colors}).draw(sh)


def _points(coords, color, size):
    if not coords:
        return
    sh = _shader('POINT_UNIFORM_COLOR')
    sh.bind()
    gpu.state.point_size_set(size)
    sh.uniform_float("color", color)
    batch_for_shader(sh, 'POINTS', {"pos": coords}).draw(sh)


def _state(context):
    sc = context.scene
    s = getattr(sc, "motion_assist", None)
    if s is None or not s.show_overlay:
        return None
    rig = s.target
    if rig is None or sc.objects.get(rig.name) is None:
        return None
    b = bodymod.get_body(rig)
    if b is None:
        return None
    data = analysis.CACHE.get(rig.name)
    f = sc.frame_current
    i = f - data["f0"] if data is not None and data["f0"] <= f <= data["f1"] else None
    return s, rig, b, data, i


def _draw_view():
    try:
        st = _state(bpy.context)
        if st is None:
            return
        s, rig, b, data, i = st
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        if data is not None and s.show_path:
            before = data.get("com_before")
            if before is not None:
                _lines([tuple(p) for p in before], [GREY] * len(before), 1.5)
            _lines([tuple(p) for p in data["com"]], [STATUS_COL[int(x)] for x in data["status"]], 2.5)
        W = b.mats(rig)
        c, _X = b.com(W)
        code = int(data["status"][i]) if i is not None else None
        col = STATUS_COL.get(code, (1.0, 1.0, 1.0, 1.0))
        if i is not None and not np.isnan(data["zg"][i]):
            zg = float(data["zg"][i])
        else:
            zg = float(b.probes_world(W)[:, 2].min()) if len(b.plocal) else 0.0
        c = tuple(float(x) for x in c)
        foot = (c[0], c[1], zg + 0.002)
        _lines([c, foot], [col, col], 1.5)
        _points([foot], col, 7)
        _points([c], col, 13)
        if i is not None:
            if s.show_support and data["hulls"][i] is not None and len(data["hulls"][i]) >= 2:
                H = [(float(x), float(y), zg + 0.003) for x, y in data["hulls"][i]]
                H.append(H[0])
                _lines(H, [(0.25, 0.9, 0.35, 0.9)] * len(H), 2.0)
            if s.show_contacts:
                P = data["P"][i][data["touch"][i]]
                _points([tuple(float(x) for x in p) for p in P], (0.3, 1.0, 0.4, 1.0), 5)
            if s.show_zmp and not np.isnan(data["zmp"][i][0]):
                x, y = (float(v) for v in data["zmp"][i])
                r = 0.035
                m = (1.0, 0.3, 0.9, 1.0)
                _lines([(x - r, y, zg + 0.004), (x + r, y, zg + 0.004), (x, y - r, zg + 0.004), (x, y + r, zg + 0.004)],
                       [m] * 4, 2.5, 'LINES')
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')
    except Exception as e:
        _errors[0] += 1
        if _errors[0] < 5:
            print("Motion Assist overlay:", e)


def _draw_text():
    try:
        st = _state(bpy.context)
        if st is None:
            return
        s, rig, b, data, i = st
        if i is not None:
            code = int(data["status"][i])
            txt = "%s: %s" % (rig.name, analysis.STATUS_NAMES[code])
            col = STATUS_COL[code]
        else:
            txt = "%s: %.0f kg (analyze to check the motion)" % (rig.name, b.M)
            col = (1.0, 1.0, 1.0, 1.0)
        blf.size(0, 15)
        blf.color(0, *col)
        blf.position(0, 70, 40, 0)
        blf.draw(0, txt)
    except Exception as e:
        _errors[0] += 1
        if _errors[0] < 5:
            print("Motion Assist text:", e)


def register():
    if not _handles:
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_view, (), 'WINDOW', 'POST_VIEW'))
        _handles.append(bpy.types.SpaceView3D.draw_handler_add(_draw_text, (), 'WINDOW', 'POST_PIXEL'))


def unregister():
    while _handles:
        bpy.types.SpaceView3D.draw_handler_remove(_handles.pop(), 'WINDOW')
