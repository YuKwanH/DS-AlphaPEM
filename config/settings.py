import numpy as np
import re
from config.initialize import parameters

solver_variable_names = ['C_H2_agc', 'C_H2_agdl', 'C_H2_acl','C_H2_mem',
                                            'C_O2_mem', 'C_O2_ccl', 'C_O2_cgdl', 'C_O2_cgc', 'C_N2',
                                            'C_v_agc', 'C_v_agdl', 'C_v_acl', 'C_v_ccl', 'C_v_cgdl', 'C_v_cgc',
                                            's_agdl', 's_acl', 's_ccl', 's_cgdl',
                                            'lambda_acl', 'lambda_ccl', 'lambda_mem',
                                            'eta_c', 'Pasm', 'Paem', 'Pcsm', 'Pcem', 'Phi_asm', 'Phi_aem', 'Phi_csm','Phi_cem',
                                            'Wcp', 'Wa_inj', 'Wc_inj', 'Abp_a', 'Abp_c',
                                            'C_Pt2_mem', 'C_Pt2_ccl', 'delta_mem', 'S_N_ccl', 'theta_ccl',
                                            "Tagdl","Tacl","Tmem","Tccl", "Tcgdl"]


# Display settings 
regions = ["agdl", "acl", "mem", "ccl", "cgdl"]
species = ["v", "O2", "H2", "s", "lambda"]
# Colormap: Temperature condition (50, 60, 70 °C)
temperature_values = [50, 60, 70]
colormap_temp = {
    50: "#000AC6",   # blue
    60: "#fa6a03",   # orange
    70: "#d70000"    # red
}

# Linemap: Pressure condition (PA = PC always equal)
pressure_values = [1.3, 1.4, 1.5]  # in bar (130, 140, 150 kPa)
linemap_pressure = {
    1.3: '-',      # solid
    1.4: '--',     # dashed
    1.5: ':'       # dotted
}

# Markermap: RH condition (0 and 0.5)
rh_values = [0, 0.5]
markermap_rh = {
    0: 'x',        # cross
    0.5: 'o'       # circle
}
# Display settings
profile_1d = {}
profile_pola = {}
borders = [-2*parameters['Hgdl']/parameters["n_gdl"],
                  0,
                  parameters['Hgdl'],
                  parameters['Hgdl'] + parameters['Hcl'], 
                  parameters['Hgdl'] + parameters['Hcl'] + parameters['Hmem'], 
                  parameters['Hgdl'] + parameters['Hcl']*2 + parameters['Hmem'], 
                  parameters['Hgdl']*2 + parameters['Hcl']*2 + parameters['Hmem'], 
                  parameters['Hgdl']*2.2 + parameters['Hcl']*2 + parameters['Hmem']]
# AGC, AGDL, ACL, MEM, CCL, CGDL, CGC
nodes = [-parameters['Hgdl']/parameters["n_gdl"]] +\
               np.linspace(borders[1], borders[2], parameters["n_gdl"]).tolist() +\
               [borders[2] + parameters['Hcl']/2] +\
                np.linspace(borders[3], borders[4], parameters["n_mem"]).tolist() +\
               [borders[4] + parameters['Hcl']/2] +\
                np.linspace(borders[5], borders[6], parameters["n_gdl"]).tolist() +\
                [borders[6]+parameters['Hgdl']/parameters["n_gdl"]]
nodes_postfix = ["agc"] + ["agdl_" + str(i+1) for i in range(parameters["n_gdl"])] + ["acl"] + ["mem_" + str(i+1) for i in range(parameters["n_mem"])] + ["ccl"] + ["cgdl_" + str(i+1) for i in range(parameters["n_gdl"])] + ["cgc"]
nodes_names_vp = [f"C_v_agdl_{i+1}" for i in range(parameters["n_gdl"])] + ["C_v_acl"] + ["C_v_ccl"] + [f"C_v_cgdl_{i+1}" for i in range(parameters["n_gdl"])]
nodes_names_H2 = [f"C_H2_agdl_{i+1}" for i in range(parameters["n_gdl"])] + ["C_H2_acl"] + [f"C_H2_mem_{i+1}" for i in range(parameters["n_mem"])]
nodes_name_O2 = [f"C_O2_mem_{i+1}" for i in range(parameters["n_mem"])] + ["C_O2_ccl"] + [f"C_O2_cgdl_{i+1}" for i in range(parameters["n_gdl"])]
nodes_names_s = [f"s_agdl_{i+1}" for i in range(parameters["n_gdl"])] + ["s_acl"] + ["s_ccl"] + [f"s_cgdl_{i+1}" for i in range(parameters["n_gdl"])]
nodes_lambda = ["lambda_acl"] + [f"lambda_mem_{i+1}" for i in range(parameters["n_mem"])] + ["lambda_ccl"]
nodes_T = ["Tagdl_" + str(i+1) for i in range(parameters["n_gdl"])] + ["Tacl"] + ["Tmem_" + str(i+1) for i in range(parameters["n_mem"])] + ["Tccl"] + ["Tcgdl_" + str(i+1) for i in range(parameters["n_gdl"])]

def has_species_value(profile_key, postfix):
    if profile_key == "v":
        return postfix.startswith("agdl_") or postfix == "acl" or postfix == "ccl" or postfix.startswith("cgdl_")
    if profile_key == "O2":
        return postfix.startswith("mem_") or postfix == "ccl" or postfix.startswith("cgdl_")
    if profile_key == "H2":
        return postfix.startswith("agdl_") or postfix == "acl" or postfix.startswith("mem_")
    if profile_key == "saturation":
        return postfix.startswith("agdl_") or postfix == "acl" or postfix == "ccl" or postfix.startswith("cgdl_")
    if profile_key == "lambda":
        return postfix == "acl" or postfix.startswith("mem_") or postfix == "ccl"
    if profile_key == "T":
        return postfix != "agc" and postfix != "cgc"
    return False

def expand_profile_on_nodes(profile_key, compact_values):
    expanded_values = []
    idx = 0
    for postfix in nodes_postfix:
        if has_species_value(profile_key, postfix):
            expanded_values.append(compact_values[idx])
            idx += 1
        else:
            expanded_values.append(0.0)
    return expanded_values

# Function to get plotting properties from condition key
def get_plot_properties(cond_key):
    """
    Extract temperature, pressure, and RH from condition key string.
    Key format: "RHA0/RHC{RH}_P{P}_T{T}"
    Returns: dict with 'color', 'linestyle', 'marker', 'label'
    """
    import re
    pattern = r'RHA\d+/RHC([\d.]+)_P([\d.]+)_T(\d+)'
    match = re.match(pattern, cond_key)

    if not match:
        return {'color': 'black', 'linestyle': '-', 'marker': 'o', 'label': cond_key}

    rh = float(match.group(1))
    p = float(match.group(2))
    t = int(match.group(3))

    return {
        'color': colormap_temp.get(t, 'black'),
        'linestyle': linemap_pressure.get(p, '-'),
        'marker': markermap_rh.get(rh, 'o'),
        'label': cond_key
    }

def plot_condition(axis, x_values, y_values, label, linewidth=1.8, markersize=5):
    temperature_match = re.search(r"T(?P<value>\d+(?:\.\d+)?)", label)
    pressure_match = re.search(r"P(?P<value>\d+(?:\.\d+)?)", label)
    humidity_matches = re.findall(r"(?:RH|HR)([AC])(?P<value>\d+(?:\.\d+)?)", label)
    temperature = float(temperature_match.group("value")) if temperature_match else None
    pressure = float(pressure_match.group("value")) if pressure_match else None
    if temperature is not None and temperature > 200:
        temperature = round(temperature - 273.15, 2)
    if pressure is not None and pressure > 20:
        pressure = round(pressure / 1000, 1) + 1
    humidity_parts = []
    for side, value_text in humidity_matches:
        value = float(value_text)
        if value > 1:
            value = value / 100
        humidity_parts.append(f"RH{side}{value:.1f}")
    humidity = "_".join(humidity_parts) if humidity_parts else None
    axis.plot(x_values, y_values, color=colormap_temp.get(temperature, "0.35"), linestyle=linemap_pressure.get(pressure, "-"), marker=markermap_rh.get(humidity, "o"), linewidth=linewidth, markersize=markersize)