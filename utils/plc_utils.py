
import numpy as np
import pdb

def get_plc_idxs(dataset):
    
    if dataset == 'TEP':

        idxs_by_plc = [
            [1, 16, 39, 41],     # XMV 1
            [2, 16, 39, 42],     # XMV 2
            [0, 16, 22, 24, 43], # XMV 3
            [3, 16, 22, 24, 44], # XMV 4
            [6, 9, 16, 46],      # XMV 6
            [11, 13, 16, 47],    # XMV 7
            [14, 16, 48],        # XMV 8
            [8, 50],             # XMV 10
            [7, 10, 51],         # XMV 11
        ]

        labels_by_plc = [
            ['D Feed', 'Stripper Underflow', 'Comp G in Product', 'D Feed (MV)'],
            ['E Feed', 'Stripper Underflow', 'Comp G in Product', 'E Feed (MV)'],
            ['A Feed', 'Stripper Underflow', 'Comp A to Reactor', 'Comp C to Reactor', 'A Feed (MV)'],
            ['A and C Feed', 'Stripper Underflow', 'Comp A to Reactor', 'Comp C to Reactor', 'A and C Feed (MV)'],
            ['Reactor Pressure', 'Purge Rate', 'Stripper Underflow', 'Purge (MV)'],
            ['Product Sep Level', 'Product Sep Underflow', 'Stripper Underflow', 'Separator (MV)'],
            ['Stripper Level', 'Stripper Underflow', 'Stripper (MV)'],
            ['Reactor Temperature', 'Reactor Coolant (MV)'],
            ['Reactor Level', 'Product Sep Temp', 'Condenser Coolant (MV)'],
        ]

    elif dataset == 'CTOWN':

        idxs_by_plc = [
            [0, 1, 2, 3, 4, 5], 		# PLC 1
            [6],                        # PLC 2
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], # PLC 3
            [24],                       # PLC 4
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], # PLC 5
            [35],                       # PLC 6
            [36],                       # PLC 7
            [37],                       # PLC 8
            [38]                        # PLC 9
        ]

        labels_by_plc = [
                ['PU1F', 'PU2F', 'J280', 'J269', 'PU1', 'PU2'],
                ['T1'],
                ['T2', 'V2F', 'J300', 'J256', 'J289', 'J415', 'J14', 'J422', 'PU4F', 'PU5F', 'PU6F', 'PU7F', 'V2', 'PU4', 'PU5', 'PU6', 'PU7'], 
                ['T3'],
                ['PU8F', 'PU10F', 'PU11F', 'J302', 'J306', 'J307', 'J317', 'PU8', 'PU10', 'PU11'],
                ['T4'],
                ['T5'],
                ['T6'],
                ['T7']
        ]

        
    elif dataset == 'SWAT':

        idxs_by_plc = [
            [0, 1, 2, 3],                   # PLC 1
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # PLC 2 without AIT201 (originally feature #5)
            [8, 15, 16, 17, 18, 19, 22],    # PLC 3 (downshifted by 1)
            [25, 26, 27, 28],               # PLC 4 (downshifted by 1)
            [33, 37, 38, 41],               # PLC 5 (downshifted by 1)
            #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # PLC 2
            # [9, 16, 17, 18, 19, 20, 23],    # PLC 3
            # [26, 27, 28, 29],               # PLC 4
            # [34, 38, 39, 42],               # PLC 5
        ]

        labels_by_plc =[
            ['FIT101', 'LIT101', 'MV101', 'P101'],
            ['AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'],
            ['MV201', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'P301'],
            ['AIT402', 'FIT401', 'LIT401', 'P401'],
            ['AIT501', 'FIT501', 'FIT502', 'P501'],
        ]
    
    else:
        print(f'Dataset "{dataset}" not found')

    return idxs_by_plc, labels_by_plc

def get_sensor_subsets_by_name(dataset):

    if dataset == 'SWAT':

        idxs_by_name = [
            [0, 1, 2, 3, 4],                   # 100s
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # 200s
            [15, 16, 17, 18, 19, 20, 21, 22, 23],    # 300s
            [24, 25, 26, 27, 28, 29, 30, 31, 32],    # 400s
            [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],  # 500s
            [46, 47, 48, 49],  # 500s
            # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # 200s
            # [16, 17, 18, 19, 20, 21, 22, 23, 24],    # 300s
            # [25, 26, 27, 28, 29, 30, 31, 32, 33],    # 400s
            # [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],  # 500s
            # [47, 48, 49, 50],  # 500s
        ]
    
        labels_by_name =[
            ['MV101', 'LIT101', 'FIT101', 'P101', 'P102'],
            ['P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'FIT201', 'AIT201', 'AIT202', 'AIT203', 'MV201'], 
            ['FIT301', 'LIT301', 'DPIT301', 'P301', 'P302', 'MV301', 'MV302', 'MV303', 'MV304'],
            ['UV401', 'P401', 'P402', 'P403', 'P404', 'AIT401', 'AIT402', 'FIT401', 'LIT401'],
            ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503'],
            ['P601', 'P602', 'P603', 'FIT601']
    ]

    elif dataset == 'TEP':

        idxs_by_name = [
            [5, 6, 7, 8, 20, 22, 23, 24, 25, 26, 27, 50],
            [9, 28, 29, 30, 31, 32, 33, 34, 35, 46],
            [10, 11, 12, 13, 36, 37, 38, 39, 40],
            [14, 15, 16, 17, 18, 48],
            [21, 47],
            [0, 1, 2, 3, 41, 42, 43, 44],
            [4, 45],
            [19],
            [49], 
            [51], 
            [52],
        ]

        labels_by_name = [
            ['Reactor Feed Rate', 'Reactor Pressure', 'Reactor Level', 'Reactor Temperature', 'Reactor Coolant Temp', 'Comp A to Reactor', 'Comp B to Reactor', 'Comp C to Reactor', 'Comp D to Reactor', 'Comp E to Reactor', 'Comp F to Reactor', 'Reactor Coolant'],
            ['Purge Rate', 'Comp A in Purge', 'Comp B in Purge', 'Comp C in Purge', 'Comp D in Purge', 'Comp E in Purge', 'Comp F in Purge', 'Comp G in Purge', 'Comp H in Purge', 'Purge'],
            ['Product Sep Temp', 'Product Sep Level', 'Product Sep Pressure', 'Product Sep Underflow', 'Comp D in Product', 'Comp E in Product', 'Comp F in Product', 'Comp G in Product', 'Comp H in Product'],
            ['Stripper Level', 'Stripper Pressure', 'Stripper Underflow', 'Stripper Temp', 'Stripper Steam Flow', 'Stripper'],
            ['Separator Coolant Temp', 'Separator'],
            ['A Feed', 'D Feed', 'E Feed', 'A and C Feed', 'D feed', 'E Feed.1', 'A Feed.1', 'A and C Feed.1'],
            ['Recycle Flow', 'Recycle'], 
            ['Compressor Work'],
            ['Steam'], 
            ['Condenser Coolant (MV)'], 
            ['Agitator'],
        ]

    # NOTE: CTOWN does not have feature names, so we just use the PLC definitions here again
    elif dataset == 'CTOWN':

        idxs_by_name = [
            [0, 1, 2, 3, 4, 5], 		# PLC 1
            [6],                        # PLC 2
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], # PLC 3
            [24],                       # PLC 4
            [25, 26, 27, 28, 29, 30, 31, 32, 33, 34], # PLC 5
            [35],                       # PLC 6
            [36],                       # PLC 7
            [37],                       # PLC 8
            [38]                        # PLC 9
        ]

        labels_by_name = [
                ['PU1F', 'PU2F', 'J280', 'J269', 'PU1', 'PU2'],
                ['T1'],
                ['T2', 'V2F', 'J300', 'J256', 'J289', 'J415', 'J14', 'J422', 'PU4F', 'PU5F', 'PU6F', 'PU7F', 'V2', 'PU4', 'PU5', 'PU6', 'PU7'], 
                ['T3'],
                ['PU8F', 'PU10F', 'PU11F', 'J302', 'J306', 'J307', 'J317', 'PU8', 'PU10', 'PU11'],
                ['T4'],
                ['T5'],
                ['T6'],
                ['T7']
        ]

    return idxs_by_name, labels_by_name

def get_fsn_process_dict(dataset):

    if dataset == "SWAT":
        process_dict = {"P1": 5, "P2": 10, "P3": 9, "P4": 9, "P5": 13, "P6": 4}
    elif dataset == "WADI":
        # Note removed 8 from P2, due to our filtering
        process_dict = {"P1": 19, "P2": 82, "P3": 15, "P4": 3}
    elif dataset == "TEP":
        process_dict = {"P1": 12, "P2": 10, "P3": 9, "P4": 6, "P5": 2, "P6": 8, "P7": 2, "P8": 4}
    elif dataset == "CTOWN":
        process_dict = {"P1": 6, "P2": 6, "P3": 12, "P4": 4, "P5": 7, "P6": 4}

    return process_dict
