MACHINE_NUM_CORES = {
    "athena":        18,
    "AMD-Desktop":   8,
    "intel-Desktop": 8,
    "zeus":          64,
    "intel-DesktopI9": 24
}

MACHINE_L1_CACHE = {
                "athena":           32 * 1024,
                "AMD-Desktop":      32 * 1024,
                "intel-Desktop":    48 * 1024,
                "zeus":             32 * 1024,
                "intel-DesktopI9":  37.3 * 1024
}

MACHINE_L2_CACHE = {
                "athena":           1024 * 1024,
                "AMD-Desktop":      512 * 1024,
                "intel-Desktop":    512 * 1024,
                "zeus":             512 * 1024,
                "intel-DesktopI9":  1365.3 * 1024
}

MACHINE_L3_CACHE = {
                "athena":           24.8 * 1024 * 1024,
                "AMD-Desktop":      32 * 1024 * 1024,
                "intel-Desktop":    16 * 1024 * 1024,
                "zeus":             256 * 1024 * 1024, # this says it has 16 instances, so it is not like other machines
                "intel-DesktopI9":  36 * 1024 * 1024
}