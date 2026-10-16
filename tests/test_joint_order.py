from dapier_act.observation.schema import CANONICAL_JOINT_ORDER


def test_joint_order_length_and_uniqueness():
    assert len(CANONICAL_JOINT_ORDER) == 12
    assert len(set(CANONICAL_JOINT_ORDER)) == 12
    assert CANONICAL_JOINT_ORDER[0].startswith("left_")
    assert CANONICAL_JOINT_ORDER[6].startswith("right_")
