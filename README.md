# Time-Dependent Fokker-Planck Equation Solver

## About the code

- `Fokker_Planck.py`: Implements a tridiagonal matrix solver for time-dependent and steady-state Fokker-Planck equations
- `test.py`: Test case with a injection rate q(p)~p exp(-p/p_inj)
- `test_inj.py`: Test case with a delta-function injection rates q(p)

## Requirements

The following Python packages are required:

- `scipy`
- `numpy` 
- `matplotlib`

## Usage

The test cases can be run directly without installation of the solver:

```bash
python test.py
```

Contact Chengchao Yuan (chengchao.yuan@desy.de) for any questions with this code.

## Citation

Please cite the paper [Coupled Time-Dependent Proton Acceleration and Leptonic-Hadronic Radiation in Turbulent Supermassive Black Hole Coronae](https://arxiv.org/abs/2508.08233) when using the code in your work.
