{
    'n_dims':       2, \
    'qbmm_type':    'CQMOM', \
    'qbmm_setup': \
    { \
        'config1d': \
        ( \
            { \
                'qbmm_type':    'QMOM', \
                'qbmm_setup': \
                { \
                    'inv_type':   'Wheeler', \
                    'inv_setup':  \
                    { \
                        'adaptive': 0, \
                        'rmin':     1e-8, \
                        'eabs':     1e-3, \
                    } \
                } \
            }, \
            { \
                'qbmm_type':    'QMOM', \
                'qbmm_setup': \
                { \
                    'inv_type':   'Wheeler', \
                    'inv_setup':  \
                    { \
                        'adaptive': 0, \
                        'rmin':     1e-8, \
                        'eabs':     1e-3, \
                    } \
                } \
            } \
        ) \
    }
}
