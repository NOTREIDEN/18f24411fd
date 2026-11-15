from pathlib import Path

MAP_ROOT=Path(__file__).resolve().parents[1]/'assets/maps'


def futsal_for(team_size):
    if team_size<=3:name=f'futsal-{max(1,team_size)}v{max(1,team_size)}.hbs'
    elif team_size<=5:name='futsal-5v5-training.hbs'
    elif team_size<=8:name='futsal-8v8-training.hbs'
    else:name='futsal-12v12-training.hbs'
    return str(MAP_ROOT/name)
