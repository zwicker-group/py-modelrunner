#!/usr/bin/env python

import tempfile
import time

import h5py
import numpy as np
import zarr


def test_h5py(path, shapes, time_points):
    f = h5py.File(path, "w")
    data = {
        key: f.create_dataset(key, (0,) + shape, maxshape=(None,) + shape, chunks=True)
        for key, shape in shapes.items()
    }
    for n in range(time_points):
        for key, shape in shapes.items():
            data[key].resize(n + 1, axis=0)
            data[key][n] = np.full(shape, n)

    for d in data.values():
        np.testing.assert_allclose(d[:, 0, 0], np.arange(time_points))


def test_zarr_zip(path, shapes, time_points):
    f = zarr.group(store=zarr.ZipStore(path, mode="w"))
    data = {
        key: f.zeros(key, shape=(0,) + shape, chunks=(1,) + shape)
        for key, shape in shapes.items()
    }
    for n in range(time_points):
        for key, shape in shapes.items():
            data[key].append([np.full(shape, n)])

    for d in data.values():
        np.testing.assert_allclose(d[:, 0, 0], np.arange(time_points))


def test_zarr_dir(path, shape, time_points):
    f = zarr.group(store=path)
    data = {
        key: f.zeros(key, shape=(0,) + shape, chunks=(1,) + shape)
        for key, shape in shapes.items()
    }
    for n in range(time_points):
        for key, shape in shapes.items():
            data[key].append([np.full(shape, n)])

    for d in data.values():
        np.testing.assert_allclose(d[:, 0, 0], np.arange(time_points))


if __name__ == "__main__":
    shapes = {"a": (64, 64), "b": (32, 32)}
    time_points = 100

    with tempfile.NamedTemporaryFile() as tmp:
        t0 = time.perf_counter()
        test_h5py(tmp.name, shapes, time_points)
        print(f"HDF: {time.perf_counter() - t0}")

    with tempfile.NamedTemporaryFile() as tmp:
        t0 = time.perf_counter()
        test_zarr_zip(tmp.name, shapes, time_points)
        print(f"zarr [zip]: {time.perf_counter() - t0}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        t0 = time.perf_counter()
        test_zarr_dir(tmpdirname, shapes, time_points)
        print(f"zarr [dir]: {time.perf_counter() - t0}")
