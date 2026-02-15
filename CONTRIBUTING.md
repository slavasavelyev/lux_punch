# Contributing to Lux Punch

Thanks for considering contributing!

## Ways to contribute

- Report bugs (with logs + environment details)
- Suggest improvements (especially around calibration & stability)
- Improve docs / add screenshots or demo GIF
- Add unit tests for math/logic and calibration
- Performance improvements (keep real-time constraints in mind)

## Development setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running tests

This repo includes lightweight unit tests for math/kinematics:

```bash
python -m unittest -v
```

## Pull request checklist

- [ ] Code builds (no syntax errors)
- [ ] Tests pass (`python -m unittest -v`)
- [ ] No breaking changes to the public UI unless discussed
- [ ] Any new behavior is documented in README

## Notes on performance

- Avoid heavy per-frame allocations inside the capture loop.
- Prefer pre-allocated buffers and simple numeric operations.
- Consider frame downscaling for pose inference (already used in the GUI app).

## License

By contributing, you agree that your contributions will be licensed under the project's license (GPL-3.0-or-later).
