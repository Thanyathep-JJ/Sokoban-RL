from sokoban_rl.sprites import WORKER, make_tiles


def test_every_tile_is_built_at_the_requested_size():
    tiles = make_tiles(48)
    for name in ("wall", "floor", "floor_shadow", "target", "crate", "crate_on", "crate_dead",
                 "worker_U", "worker_D", "worker_L", "worker_R"):
        assert tiles[name].size == (48, 48)


def test_worker_art_is_16_by_16():
    for rows in WORKER.values():
        assert len(rows) == 16 and all(len(r) == 16 for r in rows)


def test_crate_states_look_different():
    t = make_tiles(16)
    assert t["crate"].tobytes() != t["crate_on"].tobytes() != t["crate_dead"].tobytes()
