import numpy as np
import matplotlib.pyplot as plt 
import os, sys 

class ReionizationHistory:
    def __init__(self):
        self._load_constraints() 
        self._load_models()

    def _load_constraints(self):
        self.neutral_fraction_constraints = {}

        method = "Lyman-alpha dark pixel fraction"
        self.neutral_fraction_constraints["McGreer+2011"] = [
                {"redshift": 5.5, "redshift_error": [-0.5,0.0], "value": 0.20, "error": [-0.00,0.00], "type": "upper", "method": method},
                {"redshift": 6.1, "redshift_error": [-0.0,0.0], "value": 0.80, "error": [-0.00,0.00], "type": "upper", "method": method},
            ]
        self.neutral_fraction_constraints["McGreer+2015"] = [
                {"redshift": 5.6, "redshift_error": [-0.0,0.0], "value": 0.04, "error": [-0.00,0.05], "type": "upper", "method": method},
                {"redshift": 5.9, "redshift_error": [-0.0,0.0], "value": 0.06, "error": [-0.00,0.05], "type": "upper", "method": method},
            ]
        self.neutral_fraction_constraints["Jin+2023"] = [
                {"redshift": 6.3, "redshift_error": [-0.0,0.0], "value": 0.79, "error": [-0.04,0.04], "type": "upper", "method": method},
                {"redshift": 6.5, "redshift_error": [-0.0,0.0], "value": 0.87, "error": [-0.03,0.03], "type": "upper", "method": method},
                {"redshift": 6.7, "redshift_error": [-0.0,0.0], "value": 0.94, "error": [-0.09,0.06], "type": "upper", "method": method},
            ]
        
        method = "Lyman-alpha damping wing (QSOs)"
        self.neutral_fraction_constraints["Greig+2022"] = [
                {"redshift": 7.00, "redshift_error": [-0.0,0.0], "value": 0.64, "error": [-0.17,0.21], "type": "measurement", "method": method, "comment": "Reanalysis of J0252 (Wang+2020)"},
                {"redshift": 7.51, "redshift_error": [-0.0,0.0], "value": 0.27, "error": [-0.23,0.19], "type": "measurement", "method": method, "comment": "Reanalysis of J1007 (Yang+2020a)"},
                {"redshift": 7.08, "redshift_error": [-0.0,0.0], "value": 0.44, "error": [-0.24,0.23], "type": "measurement", "method": method, "comment": "Reanalysis of J1120 (Mortlock+2011; Greig+2017b), including N-V."},
                {"redshift": 7.54, "redshift_error": [-0.0,0.0], "value": 0.31, "error": [-0.19,0.18], "type": "measurement", "method": method, "comment": "Reanalysis of J1342 (Bañados+2018; Greig+2019), including N-V."},
            ]
        self.neutral_fraction_constraints["Wang+2020"] = [
                {"redshift": 7.00, "redshift_error": [-0.0,0.0], "value": 0.70, "error": [-0.23,0.20], "type": "measurement", "method": method, "comment": "J0252"},
            ]
        self.neutral_fraction_constraints["Yang+2020a"] = [
                {"redshift": 7.51, "redshift_error": [-0.0,0.0], "value": 0.39, "error": [-0.13,0.22], "type": "measurement", "method": method, "comment": "J1007"},
            ]
        self.neutral_fraction_constraints["Greig+2017b"] = [
                {"redshift": 7.08, "redshift_error": [-0.0,0.0], "value": 0.40, "error": [-0.19,0.21], "type": "measurement", "method": method, "comment": "J1120 (Mortlock et al. 2011)"},
            ]
        self.neutral_fraction_constraints["Greig+2019"] = [
                {"redshift": 7.54, "redshift_error": [-0.0,0.0], "value": 0.21, "error": [-0.19,0.17], "type": "measurement", "method": method, "comment": "J1342 (Bañados et al. 2018)"},
            ]
        
        method = "Lyman-alpha damping wing (Galaxies)"
        self.neutral_fraction_constraints["Umeda+2024"] = [
                {"redshift": 7.12, "redshift_error": [-0.08,0.06], "value": 0.53, "error": [-0.47,0.18], "type": "measurement", "method": method},
                {"redshift": 7.44, "redshift_error": [-0.24,0.34], "value": 0.65, "error": [-0.34,0.27], "type": "measurement", "method": method},
                {"redshift": 8.28, "redshift_error": [-0.44,0.41], "value": 0.91, "error": [-0.22,0.09], "type": "measurement", "method": method},
                {"redshift": 9.91, "redshift_error": [-1.15,1.49], "value": 0.92, "error": [-0.10,0.08], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Hsiao+2024"] = [
                {"redshift": 10.17, "redshift_error": [-0.0,0.0], "value": 0.90, "error": [-0.0,0.00], "type": "lower", "method": method, "comment": "MACS0647-JD"},
            ]
        self.neutral_fraction_constraints["Curtis-Lake+2023"] = [
                {"redshift": 11.48, "redshift_error": [-0.08,0.03], "value": 0.84, "error": [-0.3,0.16], "type": "measurement", "method": method, "comment": "JADES-GS-z11-0"},
            ]
        self.neutral_fraction_constraints["Mason+2025"] = [
                {"redshift": 6.5, "redshift_error": [-0.0,0.0], "value": 0.33, "error": [-0.27,0.18], "type": "measurement", "method": method},
                {"redshift": 9.3, "redshift_error": [-0.0,0.0], "value": 0.64, "error": [-0.23,0.17], "type": "measurement", "method": method},
            ]

        method = "Lyman-alpha forest"
        self.neutral_fraction_constraints["Bosman+2022"] = [
                {"redshift": 5.0, "redshift_error": [-0.0,0.0], "value": 2.446e-5, "error": [-0.051e-5,0.205e-5], "type": "measurement", "method": method},
                {"redshift": 5.1, "redshift_error": [-0.0,0.0], "value": 2.651e-5, "error": [-0.075e-5,0.129e-5], "type": "measurement", "method": method},
                {"redshift": 5.2, "redshift_error": [-0.0,0.0], "value": 2.988e-5, "error": [-0.085e-5,0.119e-5], "type": "measurement", "method": method},
                {"redshift": 5.3, "redshift_error": [-0.0,0.0], "value": 3.000e-5, "error": [-0.125e-5,0.466e-5], "type": "measurement", "method": method},
                {"redshift": 5.4, "redshift_error": [-0.0,0.0], "value": 3.498e-5, "error": [-(3.498-3.332)*1e-5,0.00], "type": "lower", "method": method},
                {"redshift": 5.5, "redshift_error": [-0.0,0.0], "value": 4.328e-5, "error": [-(4.328-4.016)*1e-5,0.00], "type": "lower", "method": method},
                {"redshift": 5.6, "redshift_error": [-0.0,0.0], "value": 5.627e-5, "error": [-(5.627-4.630)*1e-5,0.00], "type": "lower", "method": method},
                {"redshift": 5.7, "redshift_error": [-0.0,0.0], "value": 6.544e-5, "error": [-(6.544-5.990)*1e-5,0.00], "type": "lower", "method": method},
                {"redshift": 5.8, "redshift_error": [-0.0,0.0], "value": 7.087e-5, "error": [-(7.087-6.401)*1e-5,0.00], "type": "lower", "method": method},
            ]
        self.neutral_fraction_constraints["Fan+2006"] = [
                {"redshift": 5.03, "redshift_error": [-0.0,0.0], "value": 5.5e-5, "error": [-1.42e-5,1.65e-5], "type": "measurement", "method": method},
                {"redshift": 5.25, "redshift_error": [-0.0,0.0], "value": 6.7e-5, "error": [-2.07e-5,2.44e-5], "type": "measurement", "method": method},
                {"redshift": 5.45, "redshift_error": [-0.0,0.0], "value": 6.6e-5, "error": [-2.47e-5,3.01e-5], "type": "measurement", "method": method},
                {"redshift": 5.65, "redshift_error": [-0.0,0.0], "value": 8.8e-5, "error": [-3.65e-5,4.60e-5], "type": "measurement", "method": method},
                {"redshift": 5.85, "redshift_error": [-0.0,0.0], "value": 1.3e-4, "error": [-4.08e-5,4.90e-5], "type": "lower", "method": method},
                {"redshift": 6.10, "redshift_error": [-0.0,0.0], "value": 4.3e-4, "error": [-3.00e-4,3.00e-4], "type": "lower", "method": method},
            ]
        self.neutral_fraction_constraints["Yang+2020"] = [
                {"redshift": 5.4, "redshift_error": [-0.0,0.0], "value": 5.63e-5, "error": [-1.23e-5,5.63e-5], "type": "measurement", "method": method},
                {"redshift": 5.6, "redshift_error": [-0.0,0.0], "value": 7.59e-5, "error": [-6.15e-5,1.62e-5], "type": "measurement", "method": method},
                {"redshift": 5.8, "redshift_error": [-0.0,0.0], "value": 8.85e-5, "error": [-1.28e-5,1.76e-5], "type": "measurement", "method": method},
                {"redshift": 6.0, "redshift_error": [-0.0,0.0], "value": 1.13e-4, "error": [-1.88e-5,2.00e-4], "type": "lower", "method": method},
                {"redshift": 6.2, "redshift_error": [-0.0,0.0], "value": 1.03e-4, "error": [-1.06e-4,2.00e-4], "type": "lower", "method": method},
            ]
        self.neutral_fraction_constraints["Spina+2024"] = [
                {"redshift": 5.6, "redshift_error": [-0.0,0.0], "value": 0.19, "error": [-0.07,0.07], "type": "measurement", "method": method},
                {"redshift": 5.9, "redshift_error": [-0.0,0.0], "value": 0.44, "error": [-0.0,0.00], "type": "upper", "method": method},
            ]
        self.neutral_fraction_constraints["Zhu+2024"] = [
                {"redshift": 5.8, "redshift_error": [-0.2,0.2], "value": 0.061, "error": [-0.039,0.039], "type": "lower", "method": method},
            ]
        
        method = "Lyman-alpha equivalent width"
        self.neutral_fraction_constraints["Mason+2019 (68% CI)"] = [
                {"redshift": 8.0, "redshift_error": [-0.0,0.0], "value": 0.76, "error": [-0.00,0.00], "type": "lower", "method": method, "comment": r"68% credible interval"},
            ]
        self.neutral_fraction_constraints["Mason+2019 (95% CI)"] = [
                {"redshift": 8.0, "redshift_error": [-0.0,0.0], "value": 0.46, "error": [-0.00,0.00], "type": "lower", "method": method, "comment": r"95% credible interval"},
            ]
        self.neutral_fraction_constraints["Mason+2018"] = [
                {"redshift": 7.0, "redshift_error": [-0.0,0.0], "value": 0.59, "error": [-0.15,0.11], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Whitler+2020"] = [
                {"redshift": 7.0, "redshift_error": [-0.0,0.0], "value": 0.55, "error": [-0.13,0.11], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Jung+2020"] = [
                {"redshift": 7.6, "redshift_error": [-0.0,0.0], "value": 0.49, "error": [-0.19,0.19], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Bolan+2022 (68% CI)"] = [
                {"redshift": 7.6, "redshift_error": [-0.6,0.6], "value": 0.83, "error": [-0.11,0.08], "type": "measurement", "method": method, "comment": r"68% credible interval"},
                {"redshift": 6.7, "redshift_error": [-0.2,0.2], "value": 0.25, "error": [-0.00,0.00], "type": "upper", "method": method, "comment": r"68% credible interval"},
            ]
        self.neutral_fraction_constraints["Bolan+2022 (95% CI)"] = [
                {"redshift": 7.6, "redshift_error": [-0.6,0.6], "value": 0.83, "error": [-0.21,0.11], "type": "measurement", "method": method, "comment": r"95% credible interval"},
                {"redshift": 6.7, "redshift_error": [-0.2,0.2], "value": 0.44, "error": [-0.00,0.00], "type": "upper", "method": method, "comment": r"95% credible interval"},
            ]
        self.neutral_fraction_constraints["Bruton+2023"] = [
                {"redshift": 10.6, "redshift_error": [-0.0,0.0], "value": 0.88, "error": [-0.00,0.00], "type": "upper", "method": method, "comment": r"95% credible interval"},
            ]
        self.neutral_fraction_constraints["Nakane+2024"] = [
                {"redshift": 7.0, "redshift_error": [-0.0,0.0], "value": 0.79, "error": [-0.00,0.00], "type": "upper", "method": method},
                {"redshift": 8.0, "redshift_error": [-0.0,0.0], "value": 0.62, "error": [-0.36,0.15], "type": "measurement", "method": method},
                {"redshift": 10.1, "redshift_error": [-1.10,2.90], "value": 0.93, "error": [-0.07,0.04], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Tang+2024"] = [
                {"redshift": 7.0, "redshift_error": [-(7.0-6.5),(8.0-7.0)], "value": 0.48, "error": [-0.22,0.15], "type": "measurement", "method": method},
                {"redshift": 8.8, "redshift_error": [-(8.8-8.0),(10.0-8.8)], "value": 0.81, "error": [-0.24,0.12], "type": "measurement", "method": method},
                {"redshift": 11.0, "redshift_error": [-(11.0-10.0),(13.3-11.0)], "value": 0.89, "error": [-0.21,0.08], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Jones+2025"] = [
                {"redshift": 7.0, "redshift_error": [-0.5,0.5], "value": 0.64, "error": [-0.21,0.13], "type": "measurement", "method": method},
            ]
        
        method = "Lyman-alpha luminosity function"
        self.neutral_fraction_constraints["Inoue+2018"] = [
                {"redshift": 7.3, "redshift_error": [-0.0,0.0], "value": 0.5, "error": [-0.3,0.1], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Morales+2021"] = [
                {"redshift": 6.6, "redshift_error": [-0.0,0.0], "value": 0.08, "error": [-0.05,0.08], "type": "measurement", "method": method},
                {"redshift": 7.0, "redshift_error": [-0.0,0.0], "value": 0.28, "error": [-0.05,0.05], "type": "measurement", "method": method},
                {"redshift": 7.3, "redshift_error": [-0.0,0.0], "value": 0.83, "error": [-0.07,0.06], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Wold+2022"] = [
                {"redshift": 6.9, "redshift_error": [-0.0,0.0], "value": 0.33, "error": [-0.0,0.0], "type": "upper", "method": method},
            ]
        self.neutral_fraction_constraints["Umeda+2025"] = [
                {"redshift": 5.7, "redshift_error": [-0.0,0.0], "value": 0.05, "error": [-0.0,0.0], "type": "upper", "method": method},
                {"redshift": 6.6, "redshift_error": [-0.0,0.0], "value": 0.15, "error": [-0.08,0.10], "type": "measurement", "method": method},
                {"redshift": 7.0, "redshift_error": [-0.0,0.0], "value": 0.18, "error": [-0.12,0.14], "type": "measurement", "method": method},
                {"redshift": 7.3, "redshift_error": [-0.0,0.0], "value": 0.75, "error": [-0.13,0.09], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Kageura+2025"] = [
                {"redshift": 5.90, "redshift_error": [5.50-5.90,6.39-5.90], "value": 0.17, "error": [-0.16,0.23], "type": "measurement", "method": method},
                {"redshift": 6.96, "redshift_error": [6.54-6.96,7.49-6.96], "value": 0.63, "error": [-0.28,0.18], "type": "measurement", "method": method},
                {"redshift": 8.41, "redshift_error": [7.51-8.41,9.43-8.41], "value": 0.79, "error": [-0.21,0.13], "type": "measurement", "method": method},
                {"redshift": 11.0, "redshift_error": [9.62-11.0,14.18-11.], "value": 0.88, "error": [-0.13,0.11], "type": "measurement", "method": method},
            ]
        method = "Lyman-alpha clustering"
        self.neutral_fraction_constraints["Sobacchi+2015"] = [
                {"redshift": 7.0, "redshift_error": [-0.0,0.0], "value": 0.5, "error": [-0.0,0.0], "type": "upper", "method": method, "comment": r"68% credible interval"},
            ]
        self.neutral_fraction_constraints["Ouchi+2018"] = [
                {"redshift": 6.6, "redshift_error": [-0.0,0.0], "value": 0.15, "error": [-0.15,0.15], "type": "measurement", "method": method},
            ]
        self.neutral_fraction_constraints["Umeda+2025 (clustering)"] = [
                {"redshift": 5.7, "redshift_error": [-0.0,0.0], "value": 0.06, "error": [-0.03,0.12], "type": "measurement", "method": method},
                {"redshift": 6.6, "redshift_error": [-0.0,0.0], "value": 0.21, "error": [-0.14,0.19], "type": "measurement", "method": method},
            ]
        
    def _load_models(self):
        self.neutral_fraction_models = {}
        method = 'pyC2Ray'
        zs = np.array([ 5.348,  5.392,  5.436,  5.48 ,  5.526,  5.572,  5.62 ,  5.668,
                        5.717,  5.767,  5.817,  5.869,  5.922,  5.976,  6.03 ,  6.086,
                        6.143,  6.201,  6.261,  6.321,  6.383,  6.446,  6.511,  6.577,
                        6.644,  6.713,  6.784,  6.856,  6.93 ,  7.006,  7.083,  7.162,
                        7.244,  7.327,  7.412,  7.5  ,  7.59 ,  7.683,  7.778,  7.875,
                        7.976,  8.079,  8.185,  8.294,  8.407,  8.523,  8.643,  8.767,
                        8.895,  9.027,  9.163,  9.304,  9.45 ,  9.602,  9.759,  9.922,
                        10.091, 10.267, 10.451, 10.642, 11.048, 11.265, 11.492, 11.73 ,
                        11.98 , 12.242, 12.518, 12.808, 13.115, 13.44 , 13.783, 14.148,
                        14.536, 14.95 ])
        self.neutral_fraction_models["Giri+2024"] = [
                {"redshift": zs, "value": 1-np.array([1.   , 1.   , 1.   , 1.   , 1.   , 0.999, 0.999, 0.998, 0.996,
                                                    0.994, 0.992, 0.988, 0.983, 0.977, 0.969, 0.96 , 0.949, 0.936,
                                                    0.921, 0.904, 0.886, 0.865, 0.843, 0.819, 0.793, 0.766, 0.738,
                                                    0.709, 0.679, 0.648, 0.617, 0.586, 0.554, 0.523, 0.492, 0.462,
                                                    0.432, 0.402, 0.374, 0.347, 0.32 , 0.295, 0.271, 0.248, 0.226,
                                                    0.206, 0.186, 0.168, 0.152, 0.136, 0.121, 0.108, 0.095, 0.084,
                                                    0.074, 0.064, 0.055, 0.048, 0.041, 0.034, 0.024, 0.02 , 0.016,
                                                    0.013, 0.01 , 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
                                                    0.001, 0.001]), 
                                    "method": method, "name": "Source1_SinkA"},
                {"redshift": zs, "value": 1-np.array([1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   , 1.   ,
                                                    1.   , 0.999, 0.998, 0.997, 0.996, 0.994, 0.991, 0.986, 0.98 ,
                                                    0.971, 0.96 , 0.947, 0.93 , 0.911, 0.889, 0.863, 0.835, 0.805,
                                                    0.772, 0.738, 0.702, 0.665, 0.628, 0.591, 0.554, 0.518, 0.483,
                                                    0.449, 0.416, 0.385, 0.355, 0.326, 0.3  , 0.274, 0.25 , 0.228,
                                                    0.207, 0.187, 0.169, 0.152, 0.136, 0.121, 0.108, 0.095, 0.084,
                                                    0.074, 0.064, 0.055, 0.048, 0.041, 0.034, 0.024, 0.02 , 0.016,
                                                    0.013, 0.01 , 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
                                                    0.001, 0.001]), 
                                    "method": method, "name": "Source1_SinkB"},
            ]
        method = "Thesan"
        self.neutral_fraction_models["Garaldi+2022"] = [
                {"redshift": np.array([16.1067, 15.7173, 15.4543, 15.1623, 14.908 , 14.5858, 14.3014,
                                        14.6177, 14.3327, 14.0838, 13.8389, 13.5681, 13.3462, 13.1133,
                                        12.8983, 12.6726, 12.461 , 12.2865, 12.0574, 11.8585, 11.6886,
                                        11.505 , 11.3051, 11.0929, 10.9331, 10.7755, 10.5961, 10.437 ,
                                        10.2802, 10.1425, 10.0065, 10.6228, 10.4663, 10.2917, 10.1539,
                                        10.0178,  9.8834,  9.7395,  9.5976,  9.4523,  9.3194,  9.1935,
                                        9.0588,  8.9539,  8.8048,  8.7   ,  8.5964,  8.4841,  8.378 ,
                                        8.2683,  8.1763,  8.0759,  7.9904,  7.9721,  7.8875,  7.7926,
                                        7.7008,  7.6101,  7.5355,  7.4616,  7.3819,  7.3072,  7.2564,
                                        7.1828,  7.1036,  7.0293,  6.9659,  6.907 ,  6.8446,  6.7936,
                                        6.737 ,  6.6779,  6.6279,  6.5686,  6.5108,  6.461 ,  6.4097,
                                        6.3587,  6.3108,  6.2577,  6.2096,  6.16  ,  6.1125,  6.0636,
                                        6.0168,  5.9631,  5.9099,  5.8676,  5.8221,  5.7751,  5.7337,
                                        5.6873,  5.6472,  5.6023,  5.5552,  5.5118,  5.5168,  5.4587]), 
                 "value": np.array([0.99998  , 0.99957  , 0.99884  , 0.99786  , 0.99686  , 0.99541  ,
                                    0.99391  , 0.99557  , 0.99408  , 0.99257  , 0.99081  , 0.98863  ,
                                    0.98649  , 0.98392  , 0.98124  , 0.97803  , 0.97457  , 0.97119  ,
                                    0.96638  , 0.96162  , 0.95713  , 0.95165  , 0.94526  , 0.93768  ,
                                    0.93127  , 0.9243   , 0.91567  , 0.90724  , 0.89841  , 0.88969  ,
                                    0.88065  , 0.91712  , 0.90885  , 0.89906  , 0.89038  , 0.88137  ,
                                    0.872    , 0.86153  , 0.85063  , 0.83817  , 0.82599  , 0.81398  ,
                                    0.80002  , 0.78835  , 0.771    , 0.75851  , 0.74484  , 0.7293   ,
                                    0.7135   , 0.69656  , 0.68161  , 0.66454  , 0.64921  , 0.64578  ,
                                    0.62982  , 0.61091  , 0.59193  , 0.57203  , 0.55493  , 0.53699  ,
                                    0.51722  , 0.49797  , 0.48442  , 0.46432  , 0.44175  , 0.41956  ,
                                    0.39969  , 0.38045  , 0.35968  , 0.34258  , 0.32326  , 0.3029   ,
                                    0.28553  , 0.2648   , 0.24441  , 0.22669  , 0.20831  , 0.1902   ,
                                    0.17333  , 0.15498  , 0.13875  , 0.12246  , 0.10757  , 0.093002 ,
                                    0.079871 , 0.065957 , 0.053381 , 0.044333 , 0.035604 , 0.027921 ,
                                    0.022183 , 0.016721 , 0.012651 , 0.0088518, 0.005862 , 0.0038965,
                                    0.004092 , 0.0021805]), 
                                    "method": method, "name": "Thesan-1"},
        ]
        method = "Astraeus"
        self.neutral_fraction_models["Hutter+2025"] = [
                {"redshift": np.array([24.58, 24.18, 23.78, 23.38, 22.98, 22.58, 22.18, 21.78, 21.38,
                                       20.98, 20.58, 20.18, 19.78, 19.38, 18.98, 18.58, 18.18, 17.78,
                                       17.38, 16.98, 16.55, 16.12, 15.69, 15.31, 14.95, 14.6 , 14.24,
                                       13.93, 13.6 , 13.27, 12.97, 12.66, 12.35, 12.05, 11.77, 11.5 ,
                                       11.22, 10.96, 10.7 , 10.44, 10.19,  9.94,  9.71,  9.47,  9.24,
                                        9.01,  8.79,  8.58,  8.37,  8.17,  7.96,  7.76,  7.58,  7.39,
                                        7.2 ,  7.03,  6.85,  6.67,  6.51,  6.34,  6.18,  6.02,  5.87,
                                        5.72,  5.57,  5.43,  5.29,  5.15,  5.02,  4.88,  4.75,  4.63]), 
                 "value": np.array([1.00000000e+00, 9.99999999e-01, 9.99999989e-01, 9.99999955e-01,
                                    9.99999948e-01, 9.99999918e-01, 9.99999873e-01, 9.99999792e-01,
                                    9.99999611e-01, 9.99999030e-01, 9.99998120e-01, 9.99997131e-01,
                                    9.99993659e-01, 9.99989256e-01, 9.99980020e-01, 9.99961955e-01,
                                    9.99932183e-01, 9.99880943e-01, 9.99794082e-01, 9.99659002e-01,
                                    9.99434077e-01, 9.99117130e-01, 9.98656335e-01, 9.98033456e-01,
                                    9.97331825e-01, 9.96346031e-01, 9.95156165e-01, 9.93709698e-01,
                                    9.92217135e-01, 9.89834259e-01, 9.87286104e-01, 9.84567959e-01,
                                    9.80700140e-01, 9.76375121e-01, 9.71712772e-01, 9.66905593e-01,
                                    9.61269872e-01, 9.54588734e-01, 9.47890176e-01, 9.39829943e-01,
                                    9.31157350e-01, 9.21534531e-01, 9.11145616e-01, 8.99910151e-01,
                                    8.86266564e-01, 8.71077215e-01, 8.53693858e-01, 8.35219845e-01,
                                    8.13063187e-01, 7.88926242e-01, 7.58529765e-01, 7.23348934e-01,
                                    6.84594648e-01, 6.41974837e-01, 5.85098654e-01, 5.17388558e-01,
                                    4.40246461e-01, 3.25342912e-01, 2.07879026e-01, 1.03476118e-01,
                                    3.93964338e-02, 1.41376172e-02, 2.81241580e-03, 1.45386231e-04,
                                    8.01542419e-06, 7.57237981e-06, 7.70758890e-06, 7.28504671e-06,
                                    7.09302702e-06, 7.24973943e-06, 6.48938603e-06, 6.76115185e-06]), 
                                    "method": method, "name": "Evolving IMF"},
        ]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from scipy.interpolate import interp1d

    eor_hist = ReionizationHistory()
    print(eor_hist.neutral_fraction_constraints.keys())

    line_styles = [
        '-',                      # Solid
        '--',                     # Dashed
        ':',                      # Dotted
        '-.',                     # Dash-dot
        (0, (1, 1)),              # Densely dotted
        (0, (5, 1)),              # Long dash, short gap
        (0, (3, 2, 1, 2, 1, 2)),  # Dash-dot-dot
        (0, (2, 1)),              # Medium dash
        (0, (4, 4)),              # Equal dash and gap
        (0, (5, 10)),             # Long dash, long gap
        (0, (6, 2, 2, 2)),        # Double dash
        (0, (2, 2, 4, 2)),        # Alternating short/long dash
        (0, (8, 3, 3, 3)),        # Long-short-short
    ]

    constraint_list = ["McGreer+2015", "Jin+2023", "Greig+2022",
                    "Curtis-Lake+2023", "Hsiao+2024", "Umeda+2024", "Mason+2025",
                    "Yang+2020", "Bosman+2022", "Spina+2024", "Zhu+2024", #"Fan+2006",
                    "Mason+2019 (68% CI)", "Mason+2018", "Jung+2020", "Bolan+2022 (68% CI)", "Bruton+2023", "Nakane+2024", "Tang+2024", "Jones+2025",
                    "Morales+2021", "Wold+2022", "Umeda+2025", "Kageura+2025", 
                    "Sobacchi+2015", "Ouchi+2018", "Umeda+2025 (clustering)", 
                    ]
    method_list = {
        "Lyman-alpha dark pixel fraction"     : ('C0', 's', r'Ly$\alpha$ dark pixel fraction'),
        "Lyman-alpha damping wing (QSOs)"     : ('C1', '*', r'Ly$\alpha$ damping wing (QSOs)'),
        "Lyman-alpha damping wing (Galaxies)" : ('C2', 'o', r'Ly$\alpha$ damping wing (Galaxies)'),
        "Lyman-alpha forest"                  : ('C4', 'p', r'Ly$\alpha$ forest'),
        "Lyman-alpha equivalent width"        : ('C5', '8', r'Ly$\alpha$ equivalent width'),
        "Lyman-alpha luminosity function"     : ('C6', 'D', r'Ly$\alpha$ luminosity function'),
        "Lyman-alpha clustering"              : ('C7', 'X', r'Ly$\alpha$ clustering'),
    }
    model_method_list = {
        "Garaldi+2022": ('black', line_styles[2], r'Thesan-1'),
        "Giri+2024"   : ('black', line_styles[3], r'pyC$^2$Ray (Source1_SinkA)'),
        "Hutter+2025" : ('black', line_styles[7], r'Astraeus (Evolving IMF)'),
    }


    fig, ax = plt.subplots(figsize=(10, 5))

    # Constraints
    for name in constraint_list:
        dat = eor_hist.neutral_fraction_constraints[name]
        markersize = 8
        alpha = 0.75
        for neu in dat:
            color = method_list[neu['method']][0]
            marker = method_list[neu['method']][1]
            xerr = np.abs(neu['redshift_error'])[None,:].T if np.all(np.abs(neu['redshift_error'])!=0) else None
            if neu['type'] == 'measurement':
                ax.errorbar(neu['redshift'], neu['value'], yerr=np.abs(neu['error'])[None, :].T, xerr=xerr,
                            marker=marker, markersize=markersize, color=color, alpha=alpha, capsize=3, capthick=1)
            elif neu['type'] == 'upper':
                yerr = neu['error'][1] if neu['error'][1]>0.02 else 0.02
                ax.errorbar(neu['redshift'], neu['value'] + neu['error'][1], yerr=yerr, xerr=xerr, uplims=True,
                            marker=marker, markersize=markersize, color=color, alpha=alpha, capsize=3, capthick=1)
            elif neu['type'] == 'lower':
                yerr = np.abs(neu['error'][0]) if np.abs(neu['error'][0]) > 0.02 else 0.02
                ax.errorbar(neu['redshift'], neu['value'] - np.abs(neu['error'][0]), yerr=yerr, xerr=xerr, lolims=True,
                            marker=marker, markersize=markersize, color=color, alpha=alpha, capsize=3, capthick=1)

    # Dummy handles for constraint legend 
    for method, plotinfo in method_list.items():
        ax.plot([], [], label=plotinfo[2], color=plotinfo[0], marker=plotinfo[1], linestyle='')

    # Models
    for name in model_method_list:
        dat = eor_hist.neutral_fraction_models[name]
        color, lstyl, label = model_method_list[name]
        zplot = np.linspace(dat[0]['redshift'].min(), dat[0]['redshift'].max(), 50)
        fct = interp1d(dat[0]['redshift'], dat[0]['value'], kind='cubic')
        ax.plot(zplot, fct(zplot), color=color, ls=lstyl, label=None, lw=3)
        ax.plot([], [], color=color, ls=lstyl, lw=2, label=label)

    # Axes and legend 
    ax.set_xlim([5.2, 14.2])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Redshift', fontsize=18)
    ax.set_ylabel('Neutral Fraction', fontsize=18)
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=14)

    plt.tight_layout()
    # plt.savefig('reionization_history_constraints.png')
    plt.show()