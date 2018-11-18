# coding: utf-8
import eventlet


eventlet.monkey_patch(socket=True, select=True)


def spawn(func, params, size=1000):
    pool = eventlet.GreenPool(size)
    for param in params:
        pool.spawn_n(func, param)
    pool.waitall()
