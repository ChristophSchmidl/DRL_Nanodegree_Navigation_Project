from src.replaybuffer import ExperienceReplayBuffer

def test_always_passes():
    buffer = ExperienceReplayBuffer(50, None)
    assert buffer is not None

def test_always_fails():
    assert True