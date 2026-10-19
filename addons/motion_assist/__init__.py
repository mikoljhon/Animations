# Motion Assist: physics checks and fixes for character animation in Blender.
# The ideas follow what physics-based animation tools do (centre of mass, contact points, ballistic flight, spin,
# balance, overlap); every line here is our own code.

from . import body, analysis, fixes, overlap, trail, limits, autopose, grip, draw, ui

_modules = (body, analysis, fixes, overlap, trail, limits, autopose, grip, draw, ui)


def register():
    ui.register()
    draw.register()
    autopose.register()


def unregister():
    autopose.unregister()
    draw.unregister()
    ui.unregister()


def dev_reload():
    """re-import every module after the files were edited (development only)"""
    import importlib
    try:
        unregister()
    except Exception:
        pass
    for m in _modules:
        importlib.reload(m)
    register()
