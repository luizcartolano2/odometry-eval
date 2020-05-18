# Github Template

This is an API that implements a system to evaluate odometry trajectories.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 run.py gt_00.txt pred_00.txt -v
python3 run.py gt_00.txt pred_00.txt -ate
python3 run.py gt_00.txt pred_00.txt -rpe
python3 run.py gt_00.txt pred_00.txt -v -ate -rpe
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
