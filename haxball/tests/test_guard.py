from lab.guard import ProgressGuard


def report(score=.5,gd=0,reward=0,cluster=.1,loop=0):
    return {'score':score,'goal_difference':gd,'training_reward':reward,'cluster_fraction':cluster,'loop_fraction':loop,'by_format':{'1':{'score':score},'10':{'score':score}}}


def test_reward_improvement_cannot_hide_worse_score():
    g=ProgressGuard();assert g.observe(report())['promote']
    decision=g.observe(report(score=.6,gd=-1,reward=2))
    assert not decision['promote'];assert decision['intervention']=='rollback-and-reduce-style'


def test_large_format_regression_blocks_global_average_gain():
    g=ProgressGuard();g.observe(report())
    r=report(score=.7,gd=1);r['by_format']['10']['score']=.2
    assert g.observe(r)['reason']=='format-regression'


def test_plateau_requests_opponent_refresh():
    g=ProgressGuard(patience=2);g.observe(report());g.observe(report())
    assert g.observe(report())['intervention']=='refresh-opponents-and-curriculum'
