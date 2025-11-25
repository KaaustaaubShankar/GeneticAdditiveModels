# Improved Run results — seed 729

- Date (UTC): 2025-11-25T20:56:55.939073Z

- Improved GA-GAM Final Test RMSE: 0.6347

- Decision Tree Test RMSE: 0.6801

- Baseline PyGAM Test RMSE: 0.6520


## Generation Log

| Gen | Best Fitness | Average Fitness |
|---:|---:|---:|

| 0 | -0.734168 | -inf |

| 1 | -0.734168 | -inf |

| 2 | -0.691595 | -inf |

| 3 | -0.668979 | -inf |

| 4 | -0.671132 | -inf |

| 5 | -0.672176 | -inf |

| 6 | -0.663973 | -inf |

| 7 | -0.666128 | -inf |

| 8 | -0.666128 | -inf |

| 9 | -0.666128 | -inf |

| 10 | -0.666112 | -inf |

| 11 | -0.666112 | -inf |

| 12 | -0.665801 | -inf |

| 13 | -0.665573 | -inf |

| 14 | -0.665056 | -inf |

| 15 | -0.663867 | -inf |

| 16 | -0.663867 | -inf |

| 17 | -0.663728 | -inf |

| 18 | -0.663728 | -inf |

| 19 | -0.663728 | -inf |

| 20 | -0.663668 | -inf |

| 21 | -0.663650 | -inf |

| 22 | -0.663364 | -inf |

| 23 | -0.663364 | -inf |

| 24 | -0.663288 | -inf |

| 25 | -0.663286 | -inf |

| 26 | -0.663192 | -inf |

| 27 | -0.663099 | -0.669068 |

| 28 | -0.663098 | -inf |

| 29 | -0.663098 | -inf |

| 30 | -0.662937 | -inf |

| 31 | -0.662937 | -inf |

| 32 | -0.662937 | -inf |

| 33 | -0.662937 | -inf |

| 34 | -0.662937 | -inf |

| 35 | -0.662856 | -inf |

| 36 | -0.662801 | -inf |

| 37 | -0.662710 | -inf |

| 38 | -0.662164 | -inf |

| 39 | -0.661977 | -inf |

| 40 | -0.661977 | -inf |

| 41 | -0.661977 | -inf |

| 42 | -0.661966 | -inf |

| 43 | -0.661966 | -inf |

| 44 | -0.661966 | -inf |

| 45 | -0.661966 | -inf |

| 46 | -0.661966 | -inf |

| 47 | -0.661678 | -inf |

| 48 | -0.661678 | -inf |

| 49 | -0.661678 | -inf |

| 50 | -0.661669 | -inf |


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
    "knots": 9,
    "lambda": 5.551609174099606
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 7,
    "lambda": 1.9054954574794836
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 14,
    "lambda": 7.212126166100909
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 18,
    "lambda": 7.295237645207819
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
    "knots": 16,
    "lambda": 2.6659462909400538
  },
  {
    "type": "spline",
    "scale": true,
    "knots": 18,
    "lambda": 0.24711267512960947
  },
  {
    "type": "spline",
    "scale": false,
    "knots": 15,
    "lambda": 2.580724215930913
  }
]
```