# Improved Run results — seed 123

- Date (UTC): 2025-11-25T00:38:01.356278Z

- Improved GA-GAM Final Test RMSE: 0.6365

- Decision Tree Test RMSE: 0.6961

- Baseline PyGAM Test RMSE: 0.6697


## Generation Log

| Gen | Best Fitness | Average Fitness |
|---:|---:|---:|

| 0 | -0.830157 | -inf |

| 1 | -0.701255 | -inf |

| 2 | -0.689147 | -inf |

| 3 | -0.689147 | -inf |

| 4 | -0.686543 | -inf |

| 5 | -0.683132 | -inf |

| 6 | -0.681847 | -inf |

| 7 | -0.679960 | -inf |

| 8 | -0.678051 | -inf |

| 9 | -0.676709 | -inf |

| 10 | -0.676283 | -inf |

| 11 | -0.676237 | -inf |

| 12 | -0.675457 | -inf |

| 13 | -0.675426 | -inf |

| 14 | -0.675105 | -inf |

| 15 | -0.675084 | -inf |

| 16 | -0.675074 | -inf |

| 17 | -0.674910 | -inf |

| 18 | -0.674653 | -inf |

| 19 | -0.674408 | -inf |

| 20 | -0.674408 | -inf |

| 21 | -0.674183 | -inf |

| 22 | -0.674183 | -inf |

| 23 | -0.674156 | -inf |

| 24 | -0.674156 | -inf |

| 25 | -0.674156 | -inf |

| 26 | -0.674156 | -inf |

| 27 | -0.674156 | -inf |

| 28 | -0.674117 | -inf |

| 29 | -0.674117 | -inf |

| 30 | -0.674116 | -inf |

| 31 | -0.674116 | -inf |

| 32 | -0.674116 | -inf |

| 33 | -0.674116 | -inf |

| 34 | -0.673946 | -inf |

| 35 | -0.673946 | -inf |

| 36 | -0.673910 | -inf |

| 37 | -0.673910 | -inf |

| 38 | -0.673480 | -inf |

| 39 | -0.673316 | -inf |

| 40 | -0.673316 | -inf |

| 41 | -0.673316 | -inf |

| 42 | -0.673316 | -inf |

| 43 | -0.673316 | -inf |

| 44 | -0.673316 | -inf |

| 45 | -0.673316 | -inf |

| 46 | -0.673316 | -inf |

| 47 | -0.673226 | -inf |

| 48 | -0.673226 | -inf |

| 49 | -0.673216 | -inf |

| 50 | -0.673181 | -inf |

| 51 | -0.673181 | -inf |

| 52 | -0.665668 | -inf |

| 53 | -0.665668 | -inf |

| 54 | -0.658110 | -inf |

| 55 | -0.659370 | -inf |

| 56 | -0.659370 | -inf |

| 57 | -0.656639 | -inf |

| 58 | -0.654507 | -inf |

| 59 | -0.654507 | -inf |

| 60 | -0.654507 | -inf |

| 61 | -0.654507 | -inf |

| 62 | -0.654471 | -inf |

| 63 | -0.654471 | -inf |

| 64 | -0.654471 | -inf |

| 65 | -0.654471 | -inf |

| 66 | -0.654471 | -inf |

| 67 | -0.654471 | -inf |

| 68 | -0.654471 | -inf |

| 69 | -0.654471 | -inf |

| 70 | -0.654471 | -inf |

| 71 | -0.654391 | -inf |

| 72 | -0.654391 | -inf |

| 73 | -0.654391 | -0.656159 |

| 74 | -0.654341 | -inf |

| 75 | -0.654341 | -inf |

| 76 | -0.654341 | -inf |

| 77 | -0.654335 | -inf |

| 78 | -0.654335 | -inf |

| 79 | -0.654335 | -inf |

| 80 | -0.654335 | -inf |


## Model Structure Summary

- **MedInc**: spline

- **HouseAge**: spline

- **AveRooms**: spline

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
    "scale": false,
    "knots": 13,
    "lambda": 9.462562143912692
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 7,
    "lambda": 6.166666880451385
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 10,
    "lambda": 0.16655994877253086
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 13,
    "lambda": 0.37182531370338967
  },
  {
    "type": "none",
    "scale": true,
    "knots": null,
    "lambda": null
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 17,
    "lambda": 0.9878797011990934
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 18,
    "lambda": 3.0161804349258228
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 20,
    "lambda": 1.6066667457411437
  }
]
```