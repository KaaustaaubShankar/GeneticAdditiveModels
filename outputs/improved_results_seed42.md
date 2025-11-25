# Improved Run results — seed 42

- Date (UTC): 2025-11-25T19:13:22.623103Z

- Improved GA-GAM Final Test RMSE: 0.6679

- Decision Tree Test RMSE: 0.6925

- Baseline PyGAM Test RMSE: 0.7216


## Generation Log

| Gen | Best Fitness | Average Fitness |
|---:|---:|---:|

| 0 | -0.690763 | -inf |

| 1 | -0.689536 | -inf |

| 2 | -0.682781 | -inf |

| 3 | -0.681104 | -inf |

| 4 | -0.681104 | -inf |

| 5 | -0.679707 | -inf |

| 6 | -0.676150 | -inf |

| 7 | -0.675150 | -inf |

| 8 | -0.674472 | -inf |

| 9 | -0.673977 | -inf |

| 10 | -0.673786 | -inf |

| 11 | -0.673679 | -inf |

| 12 | -0.673464 | -inf |

| 13 | -0.673315 | -inf |

| 14 | -0.673313 | -inf |

| 15 | -0.673305 | -inf |

| 16 | -0.672340 | -inf |

| 17 | -0.672126 | -inf |

| 18 | -0.672126 | -inf |

| 19 | -0.672126 | -inf |

| 20 | -0.672126 | -inf |

| 21 | -0.672116 | -inf |

| 22 | -0.672110 | -inf |

| 23 | -0.672014 | -inf |

| 24 | -0.672005 | -inf |

| 25 | -0.671988 | -inf |

| 26 | -0.671955 | -inf |

| 27 | -0.671946 | -inf |

| 28 | -0.671919 | -inf |

| 29 | -0.671919 | -inf |

| 30 | -0.671919 | -inf |

| 31 | -0.671911 | -inf |

| 32 | -0.671911 | -inf |

| 33 | -0.671911 | -inf |

| 34 | -0.671904 | -inf |

| 35 | -0.671904 | -inf |

| 36 | -0.671904 | -inf |

| 37 | -0.671904 | -inf |

| 38 | -0.671896 | -inf |

| 39 | -0.671896 | -inf |

| 40 | -0.671726 | -inf |

| 41 | -0.671726 | -inf |

| 42 | -0.658607 | -0.672721 |

| 43 | -0.655998 | -inf |

| 44 | -0.655884 | -inf |

| 45 | -0.655108 | -inf |

| 46 | -0.655108 | -inf |

| 47 | -0.654991 | -inf |

| 48 | -0.654849 | -inf |

| 49 | -0.654849 | -inf |

| 50 | -0.653842 | -inf |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: spline

- **AveRooms**: none

- **AveBedrms**: spline

- **Population**: none

- **AveOccup**: spline

- **Latitude**: spline

- **Longitude**: spline

- **MedInc_baseline**: spline

- **HouseAge_baseline**: spline

- **AveRooms_baseline**: spline

- **AveBedrms_baseline**: spline

- **Population_baseline**: spline

- **AveOccup_baseline**: spline

- **Latitude_baseline**: spline

- **Longitude_baseline**: spline


## Best Chromosome (JSON)

```json
[
  {
    "type": "spline",
    "scale": true,
    "knots": 9,
    "lambda": 6.241661893366269
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 7,
    "lambda": 9.673999892103303
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 11,
    "lambda": 6.726713093004072
  },
  {
    "type": "none",
    "scale": false,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 1.2305213847166572
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 3.2509496548214853
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 20,
    "lambda": 1.216860783969543
  }
]
```