# NOTREIDEN HaxBall Lab

Research prototype: native batch simulation, recurrent policies, league evaluation and a real-room client adapter. Professional performance is an evaluation target, not an established result.

The project lives exclusively under `haxball/`; existing repository files are preserved. GitHub Actions run under repository owner **NOTREIDEN**.

Initial capacity check:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/system_probe.py
npm ci
npm run probe
```

Training/evaluation will distinguish physics ticks, agent transitions, completed episodes and full matches. Checkpoint promotion depends on heldout match performance and regression checks, never shaped reward alone.
