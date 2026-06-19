# Spatial Chromatin Model Dynamically Generated for StochPy

# --- Reactions ---

R1:
    $pool > S_free
    k_s_in
R2:
    $pool > N_free
    k_n_in
R3:
    S_free > $pool
    k_s_out * S_free
R4:
    N_free > $pool
    k_n_out * N_free
R5:
    S_free + N_free > SN_free
    k_dimerise * S_free * N_free
R6:
    SN_free > S_free + N_free
    k_dissociate * SN_free
R7:
    N_free + N_free > NN_free
    k_dimerise * 0.5 * N_free * (N_free - 1)
R8:
    NN_free > N_free + N_free
    k_dissociate * NN_free
R9:
    S_0_E + S_free > S_0_S
    k_bind_s * S_0_E * S_free
R10:
    S_0_S > S_0_E + S_free
    k_unbind_s * S_0_S
R11:
    S_0_E + N_free > S_0_N
    k_bind_n * S_0_E * N_free
R12:
    S_0_N > S_0_E + N_free
    k_unbind_n * S_0_N
R13:
    S_0_E + SN_free > S_0_SNf
    k_bind_s * S_0_E * SN_free
R14:
    S_0_SNf > S_0_E + SN_free
    k_unbind_s * S_0_SNf
R15:
    S_0_E + SN_free > S_0_NSf
    k_bind_n * S_0_E * SN_free
R16:
    S_0_NSf > S_0_E + SN_free
    k_unbind_n * S_0_NSf
R17:
    S_0_E + NN_free > S_0_NNf
    2.0 * k_bind_n * S_0_E * NN_free
R18:
    S_0_NNf > S_0_E + NN_free
    k_unbind_n * S_0_NNf
R19:
    S_0_S + N_free > S_0_SNf
    k_dimerise * S_0_S * N_free
R20:
    S_0_SNf > S_0_S + N_free
    k_dissociate * S_0_SNf
R21:
    S_0_N + S_free > S_0_NSf
    k_dimerise * S_0_N * S_free
R22:
    S_0_NSf > S_0_N + S_free
    k_dissociate * S_0_NSf
R23:
    S_0_N + N_free > S_0_NNf
    k_dimerise * S_0_N * N_free
R24:
    S_0_NNf > S_0_N + N_free
    k_dissociate * S_0_NNf
R25:
    S_1_E + S_free > S_1_S
    k_bind_s * S_1_E * S_free
R26:
    S_1_S > S_1_E + S_free
    k_unbind_s * S_1_S
R27:
    S_1_E + N_free > S_1_N
    k_bind_n * S_1_E * N_free
R28:
    S_1_N > S_1_E + N_free
    k_unbind_n * S_1_N
R29:
    S_1_E + SN_free > S_1_SNf
    k_bind_s * S_1_E * SN_free
R30:
    S_1_SNf > S_1_E + SN_free
    k_unbind_s * S_1_SNf
R31:
    S_1_E + SN_free > S_1_NSf
    k_bind_n * S_1_E * SN_free
R32:
    S_1_NSf > S_1_E + SN_free
    k_unbind_n * S_1_NSf
R33:
    S_1_E + NN_free > S_1_NNf
    2.0 * k_bind_n * S_1_E * NN_free
R34:
    S_1_NNf > S_1_E + NN_free
    k_unbind_n * S_1_NNf
R35:
    S_1_S + N_free > S_1_SNf
    k_dimerise * S_1_S * N_free
R36:
    S_1_SNf > S_1_S + N_free
    k_dissociate * S_1_SNf
R37:
    S_1_N + S_free > S_1_NSf
    k_dimerise * S_1_N * S_free
R38:
    S_1_NSf > S_1_N + S_free
    k_dissociate * S_1_NSf
R39:
    S_1_N + N_free > S_1_NNf
    k_dimerise * S_1_N * N_free
R40:
    S_1_NNf > S_1_N + N_free
    k_dissociate * S_1_NNf
R41:
    S_2_E + S_free > S_2_S
    k_bind_s * S_2_E * S_free
R42:
    S_2_S > S_2_E + S_free
    k_unbind_s * S_2_S
R43:
    S_2_E + N_free > S_2_N
    k_bind_n * S_2_E * N_free
R44:
    S_2_N > S_2_E + N_free
    k_unbind_n * S_2_N
R45:
    S_2_E + SN_free > S_2_SNf
    k_bind_s * S_2_E * SN_free
R46:
    S_2_SNf > S_2_E + SN_free
    k_unbind_s * S_2_SNf
R47:
    S_2_E + SN_free > S_2_NSf
    k_bind_n * S_2_E * SN_free
R48:
    S_2_NSf > S_2_E + SN_free
    k_unbind_n * S_2_NSf
R49:
    S_2_E + NN_free > S_2_NNf
    2.0 * k_bind_n * S_2_E * NN_free
R50:
    S_2_NNf > S_2_E + NN_free
    k_unbind_n * S_2_NNf
R51:
    S_2_S + N_free > S_2_SNf
    k_dimerise * S_2_S * N_free
R52:
    S_2_SNf > S_2_S + N_free
    k_dissociate * S_2_SNf
R53:
    S_2_N + S_free > S_2_NSf
    k_dimerise * S_2_N * S_free
R54:
    S_2_NSf > S_2_N + S_free
    k_dissociate * S_2_NSf
R55:
    S_2_N + N_free > S_2_NNf
    k_dimerise * S_2_N * N_free
R56:
    S_2_NNf > S_2_N + N_free
    k_dissociate * S_2_NNf
R57:
    S_3_E + S_free > S_3_S
    k_bind_s * S_3_E * S_free
R58:
    S_3_S > S_3_E + S_free
    k_unbind_s * S_3_S
R59:
    S_3_E + N_free > S_3_N
    k_bind_n * S_3_E * N_free
R60:
    S_3_N > S_3_E + N_free
    k_unbind_n * S_3_N
R61:
    S_3_E + SN_free > S_3_SNf
    k_bind_s * S_3_E * SN_free
R62:
    S_3_SNf > S_3_E + SN_free
    k_unbind_s * S_3_SNf
R63:
    S_3_E + SN_free > S_3_NSf
    k_bind_n * S_3_E * SN_free
R64:
    S_3_NSf > S_3_E + SN_free
    k_unbind_n * S_3_NSf
R65:
    S_3_E + NN_free > S_3_NNf
    2.0 * k_bind_n * S_3_E * NN_free
R66:
    S_3_NNf > S_3_E + NN_free
    k_unbind_n * S_3_NNf
R67:
    S_3_S + N_free > S_3_SNf
    k_dimerise * S_3_S * N_free
R68:
    S_3_SNf > S_3_S + N_free
    k_dissociate * S_3_SNf
R69:
    S_3_N + S_free > S_3_NSf
    k_dimerise * S_3_N * S_free
R70:
    S_3_NSf > S_3_N + S_free
    k_dissociate * S_3_NSf
R71:
    S_3_N + N_free > S_3_NNf
    k_dimerise * S_3_N * N_free
R72:
    S_3_NNf > S_3_N + N_free
    k_dissociate * S_3_NNf
R73:
    S_4_E + S_free > S_4_S
    k_bind_s * S_4_E * S_free
R74:
    S_4_S > S_4_E + S_free
    k_unbind_s * S_4_S
R75:
    S_4_E + N_free > S_4_N
    k_bind_n * S_4_E * N_free
R76:
    S_4_N > S_4_E + N_free
    k_unbind_n * S_4_N
R77:
    S_4_E + SN_free > S_4_SNf
    k_bind_s * S_4_E * SN_free
R78:
    S_4_SNf > S_4_E + SN_free
    k_unbind_s * S_4_SNf
R79:
    S_4_E + SN_free > S_4_NSf
    k_bind_n * S_4_E * SN_free
R80:
    S_4_NSf > S_4_E + SN_free
    k_unbind_n * S_4_NSf
R81:
    S_4_E + NN_free > S_4_NNf
    2.0 * k_bind_n * S_4_E * NN_free
R82:
    S_4_NNf > S_4_E + NN_free
    k_unbind_n * S_4_NNf
R83:
    S_4_S + N_free > S_4_SNf
    k_dimerise * S_4_S * N_free
R84:
    S_4_SNf > S_4_S + N_free
    k_dissociate * S_4_SNf
R85:
    S_4_N + S_free > S_4_NSf
    k_dimerise * S_4_N * S_free
R86:
    S_4_NSf > S_4_N + S_free
    k_dissociate * S_4_NSf
R87:
    S_4_N + N_free > S_4_NNf
    k_dimerise * S_4_N * N_free
R88:
    S_4_NNf > S_4_N + N_free
    k_dissociate * S_4_NNf
R89:
    S_5_E + S_free > S_5_S
    k_bind_s * S_5_E * S_free
R90:
    S_5_S > S_5_E + S_free
    k_unbind_s * S_5_S
R91:
    S_5_E + N_free > S_5_N
    k_bind_n * S_5_E * N_free
R92:
    S_5_N > S_5_E + N_free
    k_unbind_n * S_5_N
R93:
    S_5_E + SN_free > S_5_SNf
    k_bind_s * S_5_E * SN_free
R94:
    S_5_SNf > S_5_E + SN_free
    k_unbind_s * S_5_SNf
R95:
    S_5_E + SN_free > S_5_NSf
    k_bind_n * S_5_E * SN_free
R96:
    S_5_NSf > S_5_E + SN_free
    k_unbind_n * S_5_NSf
R97:
    S_5_E + NN_free > S_5_NNf
    2.0 * k_bind_n * S_5_E * NN_free
R98:
    S_5_NNf > S_5_E + NN_free
    k_unbind_n * S_5_NNf
R99:
    S_5_S + N_free > S_5_SNf
    k_dimerise * S_5_S * N_free
R100:
    S_5_SNf > S_5_S + N_free
    k_dissociate * S_5_SNf
R101:
    S_5_N + S_free > S_5_NSf
    k_dimerise * S_5_N * S_free
R102:
    S_5_NSf > S_5_N + S_free
    k_dissociate * S_5_NSf
R103:
    S_5_N + N_free > S_5_NNf
    k_dimerise * S_5_N * N_free
R104:
    S_5_NNf > S_5_N + N_free
    k_dissociate * S_5_NNf
R105:
    S_6_E + S_free > S_6_S
    k_bind_s * S_6_E * S_free
R106:
    S_6_S > S_6_E + S_free
    k_unbind_s * S_6_S
R107:
    S_6_E + N_free > S_6_N
    k_bind_n * S_6_E * N_free
R108:
    S_6_N > S_6_E + N_free
    k_unbind_n * S_6_N
R109:
    S_6_E + SN_free > S_6_SNf
    k_bind_s * S_6_E * SN_free
R110:
    S_6_SNf > S_6_E + SN_free
    k_unbind_s * S_6_SNf
R111:
    S_6_E + SN_free > S_6_NSf
    k_bind_n * S_6_E * SN_free
R112:
    S_6_NSf > S_6_E + SN_free
    k_unbind_n * S_6_NSf
R113:
    S_6_E + NN_free > S_6_NNf
    2.0 * k_bind_n * S_6_E * NN_free
R114:
    S_6_NNf > S_6_E + NN_free
    k_unbind_n * S_6_NNf
R115:
    S_6_S + N_free > S_6_SNf
    k_dimerise * S_6_S * N_free
R116:
    S_6_SNf > S_6_S + N_free
    k_dissociate * S_6_SNf
R117:
    S_6_N + S_free > S_6_NSf
    k_dimerise * S_6_N * S_free
R118:
    S_6_NSf > S_6_N + S_free
    k_dissociate * S_6_NSf
R119:
    S_6_N + N_free > S_6_NNf
    k_dimerise * S_6_N * N_free
R120:
    S_6_NNf > S_6_N + N_free
    k_dissociate * S_6_NNf
R121:
    S_7_E + S_free > S_7_S
    k_bind_s * S_7_E * S_free
R122:
    S_7_S > S_7_E + S_free
    k_unbind_s * S_7_S
R123:
    S_7_E + N_free > S_7_N
    k_bind_n * S_7_E * N_free
R124:
    S_7_N > S_7_E + N_free
    k_unbind_n * S_7_N
R125:
    S_7_E + SN_free > S_7_SNf
    k_bind_s * S_7_E * SN_free
R126:
    S_7_SNf > S_7_E + SN_free
    k_unbind_s * S_7_SNf
R127:
    S_7_E + SN_free > S_7_NSf
    k_bind_n * S_7_E * SN_free
R128:
    S_7_NSf > S_7_E + SN_free
    k_unbind_n * S_7_NSf
R129:
    S_7_E + NN_free > S_7_NNf
    2.0 * k_bind_n * S_7_E * NN_free
R130:
    S_7_NNf > S_7_E + NN_free
    k_unbind_n * S_7_NNf
R131:
    S_7_S + N_free > S_7_SNf
    k_dimerise * S_7_S * N_free
R132:
    S_7_SNf > S_7_S + N_free
    k_dissociate * S_7_SNf
R133:
    S_7_N + S_free > S_7_NSf
    k_dimerise * S_7_N * S_free
R134:
    S_7_NSf > S_7_N + S_free
    k_dissociate * S_7_NSf
R135:
    S_7_N + N_free > S_7_NNf
    k_dimerise * S_7_N * N_free
R136:
    S_7_NNf > S_7_N + N_free
    k_dissociate * S_7_NNf
R137:
    S_8_E + S_free > S_8_S
    k_bind_s * S_8_E * S_free
R138:
    S_8_S > S_8_E + S_free
    k_unbind_s * S_8_S
R139:
    S_8_E + N_free > S_8_N
    k_bind_n * S_8_E * N_free
R140:
    S_8_N > S_8_E + N_free
    k_unbind_n * S_8_N
R141:
    S_8_E + SN_free > S_8_SNf
    k_bind_s * S_8_E * SN_free
R142:
    S_8_SNf > S_8_E + SN_free
    k_unbind_s * S_8_SNf
R143:
    S_8_E + SN_free > S_8_NSf
    k_bind_n * S_8_E * SN_free
R144:
    S_8_NSf > S_8_E + SN_free
    k_unbind_n * S_8_NSf
R145:
    S_8_E + NN_free > S_8_NNf
    2.0 * k_bind_n * S_8_E * NN_free
R146:
    S_8_NNf > S_8_E + NN_free
    k_unbind_n * S_8_NNf
R147:
    S_8_S + N_free > S_8_SNf
    k_dimerise * S_8_S * N_free
R148:
    S_8_SNf > S_8_S + N_free
    k_dissociate * S_8_SNf
R149:
    S_8_N + S_free > S_8_NSf
    k_dimerise * S_8_N * S_free
R150:
    S_8_NSf > S_8_N + S_free
    k_dissociate * S_8_NSf
R151:
    S_8_N + N_free > S_8_NNf
    k_dimerise * S_8_N * N_free
R152:
    S_8_NNf > S_8_N + N_free
    k_dissociate * S_8_NNf
R153:
    S_9_E + S_free > S_9_S
    k_bind_s * S_9_E * S_free
R154:
    S_9_S > S_9_E + S_free
    k_unbind_s * S_9_S
R155:
    S_9_E + N_free > S_9_N
    k_bind_n * S_9_E * N_free
R156:
    S_9_N > S_9_E + N_free
    k_unbind_n * S_9_N
R157:
    S_9_E + SN_free > S_9_SNf
    k_bind_s * S_9_E * SN_free
R158:
    S_9_SNf > S_9_E + SN_free
    k_unbind_s * S_9_SNf
R159:
    S_9_E + SN_free > S_9_NSf
    k_bind_n * S_9_E * SN_free
R160:
    S_9_NSf > S_9_E + SN_free
    k_unbind_n * S_9_NSf
R161:
    S_9_E + NN_free > S_9_NNf
    2.0 * k_bind_n * S_9_E * NN_free
R162:
    S_9_NNf > S_9_E + NN_free
    k_unbind_n * S_9_NNf
R163:
    S_9_S + N_free > S_9_SNf
    k_dimerise * S_9_S * N_free
R164:
    S_9_SNf > S_9_S + N_free
    k_dissociate * S_9_SNf
R165:
    S_9_N + S_free > S_9_NSf
    k_dimerise * S_9_N * S_free
R166:
    S_9_NSf > S_9_N + S_free
    k_dissociate * S_9_NSf
R167:
    S_9_N + N_free > S_9_NNf
    k_dimerise * S_9_N * N_free
R168:
    S_9_NNf > S_9_N + N_free
    k_dissociate * S_9_NNf
R169:
    S_0_S + S_1_N > T_0_1_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_1_N
R170:
    T_0_1_SN > S_0_S + S_1_N
    k_dissociate * T_0_1_SN
R171:
    S_0_N + S_1_S > T_0_1_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_1_S
R172:
    T_0_1_NS > S_0_N + S_1_S
    k_dissociate * T_0_1_NS
R173:
    S_0_N + S_1_N > T_0_1_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_1_N * 2.0
R174:
    T_0_1_NN > S_0_N + S_1_N
    k_dissociate * T_0_1_NN
R175:
    S_0_SNf + S_1_E > T_0_1_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_1_E
R176:
    T_0_1_SN > S_0_SNf + S_1_E
    k_unbind_n * T_0_1_SN
R177:
    S_0_NSf + S_1_E > T_0_1_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_1_E
R178:
    T_0_1_NS > S_0_NSf + S_1_E
    k_unbind_s * T_0_1_NS
R179:
    S_0_NNf + S_1_E > T_0_1_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_1_E
R180:
    T_0_1_NN > S_0_NNf + S_1_E
    k_unbind_n * T_0_1_NN
R181:
    S_0_S + S_2_N > T_0_2_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_2_N
R182:
    T_0_2_SN > S_0_S + S_2_N
    k_dissociate * T_0_2_SN
R183:
    S_0_N + S_2_S > T_0_2_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_2_S
R184:
    T_0_2_NS > S_0_N + S_2_S
    k_dissociate * T_0_2_NS
R185:
    S_0_N + S_2_N > T_0_2_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_2_N * 2.0
R186:
    T_0_2_NN > S_0_N + S_2_N
    k_dissociate * T_0_2_NN
R187:
    S_0_SNf + S_2_E > T_0_2_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_2_E
R188:
    T_0_2_SN > S_0_SNf + S_2_E
    k_unbind_n * T_0_2_SN
R189:
    S_0_NSf + S_2_E > T_0_2_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_2_E
R190:
    T_0_2_NS > S_0_NSf + S_2_E
    k_unbind_s * T_0_2_NS
R191:
    S_0_NNf + S_2_E > T_0_2_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_2_E
R192:
    T_0_2_NN > S_0_NNf + S_2_E
    k_unbind_n * T_0_2_NN
R193:
    S_0_S + S_3_N > T_0_3_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_3_N
R194:
    T_0_3_SN > S_0_S + S_3_N
    k_dissociate * T_0_3_SN
R195:
    S_0_N + S_3_S > T_0_3_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_3_S
R196:
    T_0_3_NS > S_0_N + S_3_S
    k_dissociate * T_0_3_NS
R197:
    S_0_N + S_3_N > T_0_3_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_3_N * 2.0
R198:
    T_0_3_NN > S_0_N + S_3_N
    k_dissociate * T_0_3_NN
R199:
    S_0_SNf + S_3_E > T_0_3_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_3_E
R200:
    T_0_3_SN > S_0_SNf + S_3_E
    k_unbind_n * T_0_3_SN
R201:
    S_0_NSf + S_3_E > T_0_3_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_3_E
R202:
    T_0_3_NS > S_0_NSf + S_3_E
    k_unbind_s * T_0_3_NS
R203:
    S_0_NNf + S_3_E > T_0_3_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_3_E
R204:
    T_0_3_NN > S_0_NNf + S_3_E
    k_unbind_n * T_0_3_NN
R205:
    S_0_S + S_4_N > T_0_4_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_4_N
R206:
    T_0_4_SN > S_0_S + S_4_N
    k_dissociate * T_0_4_SN
R207:
    S_0_N + S_4_S > T_0_4_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_4_S
R208:
    T_0_4_NS > S_0_N + S_4_S
    k_dissociate * T_0_4_NS
R209:
    S_0_N + S_4_N > T_0_4_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_4_N * 2.0
R210:
    T_0_4_NN > S_0_N + S_4_N
    k_dissociate * T_0_4_NN
R211:
    S_0_SNf + S_4_E > T_0_4_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_4_E
R212:
    T_0_4_SN > S_0_SNf + S_4_E
    k_unbind_n * T_0_4_SN
R213:
    S_0_NSf + S_4_E > T_0_4_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_4_E
R214:
    T_0_4_NS > S_0_NSf + S_4_E
    k_unbind_s * T_0_4_NS
R215:
    S_0_NNf + S_4_E > T_0_4_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_4_E
R216:
    T_0_4_NN > S_0_NNf + S_4_E
    k_unbind_n * T_0_4_NN
R217:
    S_0_S + S_5_N > T_0_5_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_5_N
R218:
    T_0_5_SN > S_0_S + S_5_N
    k_dissociate * T_0_5_SN
R219:
    S_0_N + S_5_S > T_0_5_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_5_S
R220:
    T_0_5_NS > S_0_N + S_5_S
    k_dissociate * T_0_5_NS
R221:
    S_0_N + S_5_N > T_0_5_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_5_N * 2.0
R222:
    T_0_5_NN > S_0_N + S_5_N
    k_dissociate * T_0_5_NN
R223:
    S_0_SNf + S_5_E > T_0_5_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_5_E
R224:
    T_0_5_SN > S_0_SNf + S_5_E
    k_unbind_n * T_0_5_SN
R225:
    S_0_NSf + S_5_E > T_0_5_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_5_E
R226:
    T_0_5_NS > S_0_NSf + S_5_E
    k_unbind_s * T_0_5_NS
R227:
    S_0_NNf + S_5_E > T_0_5_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_5_E
R228:
    T_0_5_NN > S_0_NNf + S_5_E
    k_unbind_n * T_0_5_NN
R229:
    S_0_S + S_6_N > T_0_6_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_6_N
R230:
    T_0_6_SN > S_0_S + S_6_N
    k_dissociate * T_0_6_SN
R231:
    S_0_N + S_6_S > T_0_6_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_6_S
R232:
    T_0_6_NS > S_0_N + S_6_S
    k_dissociate * T_0_6_NS
R233:
    S_0_N + S_6_N > T_0_6_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_6_N * 2.0
R234:
    T_0_6_NN > S_0_N + S_6_N
    k_dissociate * T_0_6_NN
R235:
    S_0_SNf + S_6_E > T_0_6_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_6_E
R236:
    T_0_6_SN > S_0_SNf + S_6_E
    k_unbind_n * T_0_6_SN
R237:
    S_0_NSf + S_6_E > T_0_6_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_6_E
R238:
    T_0_6_NS > S_0_NSf + S_6_E
    k_unbind_s * T_0_6_NS
R239:
    S_0_NNf + S_6_E > T_0_6_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_6_E
R240:
    T_0_6_NN > S_0_NNf + S_6_E
    k_unbind_n * T_0_6_NN
R241:
    S_0_S + S_7_N > T_0_7_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_7_N
R242:
    T_0_7_SN > S_0_S + S_7_N
    k_dissociate * T_0_7_SN
R243:
    S_0_N + S_7_S > T_0_7_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_7_S
R244:
    T_0_7_NS > S_0_N + S_7_S
    k_dissociate * T_0_7_NS
R245:
    S_0_N + S_7_N > T_0_7_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_7_N * 2.0
R246:
    T_0_7_NN > S_0_N + S_7_N
    k_dissociate * T_0_7_NN
R247:
    S_0_SNf + S_7_E > T_0_7_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_7_E
R248:
    T_0_7_SN > S_0_SNf + S_7_E
    k_unbind_n * T_0_7_SN
R249:
    S_0_NSf + S_7_E > T_0_7_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_7_E
R250:
    T_0_7_NS > S_0_NSf + S_7_E
    k_unbind_s * T_0_7_NS
R251:
    S_0_NNf + S_7_E > T_0_7_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_7_E
R252:
    T_0_7_NN > S_0_NNf + S_7_E
    k_unbind_n * T_0_7_NN
R253:
    S_0_S + S_8_N > T_0_8_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_8_N
R254:
    T_0_8_SN > S_0_S + S_8_N
    k_dissociate * T_0_8_SN
R255:
    S_0_N + S_8_S > T_0_8_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_8_S
R256:
    T_0_8_NS > S_0_N + S_8_S
    k_dissociate * T_0_8_NS
R257:
    S_0_N + S_8_N > T_0_8_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_8_N * 2.0
R258:
    T_0_8_NN > S_0_N + S_8_N
    k_dissociate * T_0_8_NN
R259:
    S_0_SNf + S_8_E > T_0_8_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_8_E
R260:
    T_0_8_SN > S_0_SNf + S_8_E
    k_unbind_n * T_0_8_SN
R261:
    S_0_NSf + S_8_E > T_0_8_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_8_E
R262:
    T_0_8_NS > S_0_NSf + S_8_E
    k_unbind_s * T_0_8_NS
R263:
    S_0_NNf + S_8_E > T_0_8_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_8_E
R264:
    T_0_8_NN > S_0_NNf + S_8_E
    k_unbind_n * T_0_8_NN
R265:
    S_0_S + S_9_N > T_0_9_SN
    k_dimerise * 0.1111111111111111 * S_0_S * S_9_N
R266:
    T_0_9_SN > S_0_S + S_9_N
    k_dissociate * T_0_9_SN
R267:
    S_0_N + S_9_S > T_0_9_NS
    k_dimerise * 0.1111111111111111 * S_0_N * S_9_S
R268:
    T_0_9_NS > S_0_N + S_9_S
    k_dissociate * T_0_9_NS
R269:
    S_0_N + S_9_N > T_0_9_NN
    k_dimerise * 0.1111111111111111 * S_0_N * S_9_N * 2.0
R270:
    T_0_9_NN > S_0_N + S_9_N
    k_dissociate * T_0_9_NN
R271:
    S_0_SNf + S_9_E > T_0_9_SN
    k_bind_n * 0.1111111111111111 * S_0_SNf * S_9_E
R272:
    T_0_9_SN > S_0_SNf + S_9_E
    k_unbind_n * T_0_9_SN
R273:
    S_0_NSf + S_9_E > T_0_9_NS
    k_bind_s * 0.1111111111111111 * S_0_NSf * S_9_E
R274:
    T_0_9_NS > S_0_NSf + S_9_E
    k_unbind_s * T_0_9_NS
R275:
    S_0_NNf + S_9_E > T_0_9_NN
    k_bind_n * 0.1111111111111111 * S_0_NNf * S_9_E
R276:
    T_0_9_NN > S_0_NNf + S_9_E
    k_unbind_n * T_0_9_NN
R277:
    S_1_SNf + S_0_E > T_0_1_NS
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_0_E
R278:
    T_0_1_NS > S_1_SNf + S_0_E
    k_unbind_n * T_0_1_NS
R279:
    S_1_NSf + S_0_E > T_0_1_SN
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_0_E
R280:
    T_0_1_SN > S_1_NSf + S_0_E
    k_unbind_s * T_0_1_SN
R281:
    S_1_NNf + S_0_E > T_0_1_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_0_E
R282:
    T_0_1_NN > S_1_NNf + S_0_E
    k_unbind_n * T_0_1_NN
R283:
    S_1_S + S_2_N > T_1_2_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_2_N
R284:
    T_1_2_SN > S_1_S + S_2_N
    k_dissociate * T_1_2_SN
R285:
    S_1_N + S_2_S > T_1_2_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_2_S
R286:
    T_1_2_NS > S_1_N + S_2_S
    k_dissociate * T_1_2_NS
R287:
    S_1_N + S_2_N > T_1_2_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_2_N * 2.0
R288:
    T_1_2_NN > S_1_N + S_2_N
    k_dissociate * T_1_2_NN
R289:
    S_1_SNf + S_2_E > T_1_2_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_2_E
R290:
    T_1_2_SN > S_1_SNf + S_2_E
    k_unbind_n * T_1_2_SN
R291:
    S_1_NSf + S_2_E > T_1_2_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_2_E
R292:
    T_1_2_NS > S_1_NSf + S_2_E
    k_unbind_s * T_1_2_NS
R293:
    S_1_NNf + S_2_E > T_1_2_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_2_E
R294:
    T_1_2_NN > S_1_NNf + S_2_E
    k_unbind_n * T_1_2_NN
R295:
    S_1_S + S_3_N > T_1_3_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_3_N
R296:
    T_1_3_SN > S_1_S + S_3_N
    k_dissociate * T_1_3_SN
R297:
    S_1_N + S_3_S > T_1_3_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_3_S
R298:
    T_1_3_NS > S_1_N + S_3_S
    k_dissociate * T_1_3_NS
R299:
    S_1_N + S_3_N > T_1_3_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_3_N * 2.0
R300:
    T_1_3_NN > S_1_N + S_3_N
    k_dissociate * T_1_3_NN
R301:
    S_1_SNf + S_3_E > T_1_3_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_3_E
R302:
    T_1_3_SN > S_1_SNf + S_3_E
    k_unbind_n * T_1_3_SN
R303:
    S_1_NSf + S_3_E > T_1_3_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_3_E
R304:
    T_1_3_NS > S_1_NSf + S_3_E
    k_unbind_s * T_1_3_NS
R305:
    S_1_NNf + S_3_E > T_1_3_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_3_E
R306:
    T_1_3_NN > S_1_NNf + S_3_E
    k_unbind_n * T_1_3_NN
R307:
    S_1_S + S_4_N > T_1_4_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_4_N
R308:
    T_1_4_SN > S_1_S + S_4_N
    k_dissociate * T_1_4_SN
R309:
    S_1_N + S_4_S > T_1_4_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_4_S
R310:
    T_1_4_NS > S_1_N + S_4_S
    k_dissociate * T_1_4_NS
R311:
    S_1_N + S_4_N > T_1_4_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_4_N * 2.0
R312:
    T_1_4_NN > S_1_N + S_4_N
    k_dissociate * T_1_4_NN
R313:
    S_1_SNf + S_4_E > T_1_4_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_4_E
R314:
    T_1_4_SN > S_1_SNf + S_4_E
    k_unbind_n * T_1_4_SN
R315:
    S_1_NSf + S_4_E > T_1_4_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_4_E
R316:
    T_1_4_NS > S_1_NSf + S_4_E
    k_unbind_s * T_1_4_NS
R317:
    S_1_NNf + S_4_E > T_1_4_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_4_E
R318:
    T_1_4_NN > S_1_NNf + S_4_E
    k_unbind_n * T_1_4_NN
R319:
    S_1_S + S_5_N > T_1_5_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_5_N
R320:
    T_1_5_SN > S_1_S + S_5_N
    k_dissociate * T_1_5_SN
R321:
    S_1_N + S_5_S > T_1_5_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_5_S
R322:
    T_1_5_NS > S_1_N + S_5_S
    k_dissociate * T_1_5_NS
R323:
    S_1_N + S_5_N > T_1_5_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_5_N * 2.0
R324:
    T_1_5_NN > S_1_N + S_5_N
    k_dissociate * T_1_5_NN
R325:
    S_1_SNf + S_5_E > T_1_5_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_5_E
R326:
    T_1_5_SN > S_1_SNf + S_5_E
    k_unbind_n * T_1_5_SN
R327:
    S_1_NSf + S_5_E > T_1_5_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_5_E
R328:
    T_1_5_NS > S_1_NSf + S_5_E
    k_unbind_s * T_1_5_NS
R329:
    S_1_NNf + S_5_E > T_1_5_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_5_E
R330:
    T_1_5_NN > S_1_NNf + S_5_E
    k_unbind_n * T_1_5_NN
R331:
    S_1_S + S_6_N > T_1_6_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_6_N
R332:
    T_1_6_SN > S_1_S + S_6_N
    k_dissociate * T_1_6_SN
R333:
    S_1_N + S_6_S > T_1_6_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_6_S
R334:
    T_1_6_NS > S_1_N + S_6_S
    k_dissociate * T_1_6_NS
R335:
    S_1_N + S_6_N > T_1_6_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_6_N * 2.0
R336:
    T_1_6_NN > S_1_N + S_6_N
    k_dissociate * T_1_6_NN
R337:
    S_1_SNf + S_6_E > T_1_6_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_6_E
R338:
    T_1_6_SN > S_1_SNf + S_6_E
    k_unbind_n * T_1_6_SN
R339:
    S_1_NSf + S_6_E > T_1_6_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_6_E
R340:
    T_1_6_NS > S_1_NSf + S_6_E
    k_unbind_s * T_1_6_NS
R341:
    S_1_NNf + S_6_E > T_1_6_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_6_E
R342:
    T_1_6_NN > S_1_NNf + S_6_E
    k_unbind_n * T_1_6_NN
R343:
    S_1_S + S_7_N > T_1_7_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_7_N
R344:
    T_1_7_SN > S_1_S + S_7_N
    k_dissociate * T_1_7_SN
R345:
    S_1_N + S_7_S > T_1_7_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_7_S
R346:
    T_1_7_NS > S_1_N + S_7_S
    k_dissociate * T_1_7_NS
R347:
    S_1_N + S_7_N > T_1_7_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_7_N * 2.0
R348:
    T_1_7_NN > S_1_N + S_7_N
    k_dissociate * T_1_7_NN
R349:
    S_1_SNf + S_7_E > T_1_7_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_7_E
R350:
    T_1_7_SN > S_1_SNf + S_7_E
    k_unbind_n * T_1_7_SN
R351:
    S_1_NSf + S_7_E > T_1_7_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_7_E
R352:
    T_1_7_NS > S_1_NSf + S_7_E
    k_unbind_s * T_1_7_NS
R353:
    S_1_NNf + S_7_E > T_1_7_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_7_E
R354:
    T_1_7_NN > S_1_NNf + S_7_E
    k_unbind_n * T_1_7_NN
R355:
    S_1_S + S_8_N > T_1_8_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_8_N
R356:
    T_1_8_SN > S_1_S + S_8_N
    k_dissociate * T_1_8_SN
R357:
    S_1_N + S_8_S > T_1_8_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_8_S
R358:
    T_1_8_NS > S_1_N + S_8_S
    k_dissociate * T_1_8_NS
R359:
    S_1_N + S_8_N > T_1_8_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_8_N * 2.0
R360:
    T_1_8_NN > S_1_N + S_8_N
    k_dissociate * T_1_8_NN
R361:
    S_1_SNf + S_8_E > T_1_8_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_8_E
R362:
    T_1_8_SN > S_1_SNf + S_8_E
    k_unbind_n * T_1_8_SN
R363:
    S_1_NSf + S_8_E > T_1_8_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_8_E
R364:
    T_1_8_NS > S_1_NSf + S_8_E
    k_unbind_s * T_1_8_NS
R365:
    S_1_NNf + S_8_E > T_1_8_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_8_E
R366:
    T_1_8_NN > S_1_NNf + S_8_E
    k_unbind_n * T_1_8_NN
R367:
    S_1_S + S_9_N > T_1_9_SN
    k_dimerise * 0.1111111111111111 * S_1_S * S_9_N
R368:
    T_1_9_SN > S_1_S + S_9_N
    k_dissociate * T_1_9_SN
R369:
    S_1_N + S_9_S > T_1_9_NS
    k_dimerise * 0.1111111111111111 * S_1_N * S_9_S
R370:
    T_1_9_NS > S_1_N + S_9_S
    k_dissociate * T_1_9_NS
R371:
    S_1_N + S_9_N > T_1_9_NN
    k_dimerise * 0.1111111111111111 * S_1_N * S_9_N * 2.0
R372:
    T_1_9_NN > S_1_N + S_9_N
    k_dissociate * T_1_9_NN
R373:
    S_1_SNf + S_9_E > T_1_9_SN
    k_bind_n * 0.1111111111111111 * S_1_SNf * S_9_E
R374:
    T_1_9_SN > S_1_SNf + S_9_E
    k_unbind_n * T_1_9_SN
R375:
    S_1_NSf + S_9_E > T_1_9_NS
    k_bind_s * 0.1111111111111111 * S_1_NSf * S_9_E
R376:
    T_1_9_NS > S_1_NSf + S_9_E
    k_unbind_s * T_1_9_NS
R377:
    S_1_NNf + S_9_E > T_1_9_NN
    k_bind_n * 0.1111111111111111 * S_1_NNf * S_9_E
R378:
    T_1_9_NN > S_1_NNf + S_9_E
    k_unbind_n * T_1_9_NN
R379:
    S_2_SNf + S_0_E > T_0_2_NS
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_0_E
R380:
    T_0_2_NS > S_2_SNf + S_0_E
    k_unbind_n * T_0_2_NS
R381:
    S_2_NSf + S_0_E > T_0_2_SN
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_0_E
R382:
    T_0_2_SN > S_2_NSf + S_0_E
    k_unbind_s * T_0_2_SN
R383:
    S_2_NNf + S_0_E > T_0_2_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_0_E
R384:
    T_0_2_NN > S_2_NNf + S_0_E
    k_unbind_n * T_0_2_NN
R385:
    S_2_SNf + S_1_E > T_1_2_NS
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_1_E
R386:
    T_1_2_NS > S_2_SNf + S_1_E
    k_unbind_n * T_1_2_NS
R387:
    S_2_NSf + S_1_E > T_1_2_SN
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_1_E
R388:
    T_1_2_SN > S_2_NSf + S_1_E
    k_unbind_s * T_1_2_SN
R389:
    S_2_NNf + S_1_E > T_1_2_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_1_E
R390:
    T_1_2_NN > S_2_NNf + S_1_E
    k_unbind_n * T_1_2_NN
R391:
    S_2_S + S_3_N > T_2_3_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_3_N
R392:
    T_2_3_SN > S_2_S + S_3_N
    k_dissociate * T_2_3_SN
R393:
    S_2_N + S_3_S > T_2_3_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_3_S
R394:
    T_2_3_NS > S_2_N + S_3_S
    k_dissociate * T_2_3_NS
R395:
    S_2_N + S_3_N > T_2_3_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_3_N * 2.0
R396:
    T_2_3_NN > S_2_N + S_3_N
    k_dissociate * T_2_3_NN
R397:
    S_2_SNf + S_3_E > T_2_3_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_3_E
R398:
    T_2_3_SN > S_2_SNf + S_3_E
    k_unbind_n * T_2_3_SN
R399:
    S_2_NSf + S_3_E > T_2_3_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_3_E
R400:
    T_2_3_NS > S_2_NSf + S_3_E
    k_unbind_s * T_2_3_NS
R401:
    S_2_NNf + S_3_E > T_2_3_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_3_E
R402:
    T_2_3_NN > S_2_NNf + S_3_E
    k_unbind_n * T_2_3_NN
R403:
    S_2_S + S_4_N > T_2_4_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_4_N
R404:
    T_2_4_SN > S_2_S + S_4_N
    k_dissociate * T_2_4_SN
R405:
    S_2_N + S_4_S > T_2_4_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_4_S
R406:
    T_2_4_NS > S_2_N + S_4_S
    k_dissociate * T_2_4_NS
R407:
    S_2_N + S_4_N > T_2_4_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_4_N * 2.0
R408:
    T_2_4_NN > S_2_N + S_4_N
    k_dissociate * T_2_4_NN
R409:
    S_2_SNf + S_4_E > T_2_4_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_4_E
R410:
    T_2_4_SN > S_2_SNf + S_4_E
    k_unbind_n * T_2_4_SN
R411:
    S_2_NSf + S_4_E > T_2_4_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_4_E
R412:
    T_2_4_NS > S_2_NSf + S_4_E
    k_unbind_s * T_2_4_NS
R413:
    S_2_NNf + S_4_E > T_2_4_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_4_E
R414:
    T_2_4_NN > S_2_NNf + S_4_E
    k_unbind_n * T_2_4_NN
R415:
    S_2_S + S_5_N > T_2_5_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_5_N
R416:
    T_2_5_SN > S_2_S + S_5_N
    k_dissociate * T_2_5_SN
R417:
    S_2_N + S_5_S > T_2_5_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_5_S
R418:
    T_2_5_NS > S_2_N + S_5_S
    k_dissociate * T_2_5_NS
R419:
    S_2_N + S_5_N > T_2_5_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_5_N * 2.0
R420:
    T_2_5_NN > S_2_N + S_5_N
    k_dissociate * T_2_5_NN
R421:
    S_2_SNf + S_5_E > T_2_5_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_5_E
R422:
    T_2_5_SN > S_2_SNf + S_5_E
    k_unbind_n * T_2_5_SN
R423:
    S_2_NSf + S_5_E > T_2_5_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_5_E
R424:
    T_2_5_NS > S_2_NSf + S_5_E
    k_unbind_s * T_2_5_NS
R425:
    S_2_NNf + S_5_E > T_2_5_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_5_E
R426:
    T_2_5_NN > S_2_NNf + S_5_E
    k_unbind_n * T_2_5_NN
R427:
    S_2_S + S_6_N > T_2_6_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_6_N
R428:
    T_2_6_SN > S_2_S + S_6_N
    k_dissociate * T_2_6_SN
R429:
    S_2_N + S_6_S > T_2_6_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_6_S
R430:
    T_2_6_NS > S_2_N + S_6_S
    k_dissociate * T_2_6_NS
R431:
    S_2_N + S_6_N > T_2_6_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_6_N * 2.0
R432:
    T_2_6_NN > S_2_N + S_6_N
    k_dissociate * T_2_6_NN
R433:
    S_2_SNf + S_6_E > T_2_6_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_6_E
R434:
    T_2_6_SN > S_2_SNf + S_6_E
    k_unbind_n * T_2_6_SN
R435:
    S_2_NSf + S_6_E > T_2_6_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_6_E
R436:
    T_2_6_NS > S_2_NSf + S_6_E
    k_unbind_s * T_2_6_NS
R437:
    S_2_NNf + S_6_E > T_2_6_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_6_E
R438:
    T_2_6_NN > S_2_NNf + S_6_E
    k_unbind_n * T_2_6_NN
R439:
    S_2_S + S_7_N > T_2_7_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_7_N
R440:
    T_2_7_SN > S_2_S + S_7_N
    k_dissociate * T_2_7_SN
R441:
    S_2_N + S_7_S > T_2_7_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_7_S
R442:
    T_2_7_NS > S_2_N + S_7_S
    k_dissociate * T_2_7_NS
R443:
    S_2_N + S_7_N > T_2_7_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_7_N * 2.0
R444:
    T_2_7_NN > S_2_N + S_7_N
    k_dissociate * T_2_7_NN
R445:
    S_2_SNf + S_7_E > T_2_7_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_7_E
R446:
    T_2_7_SN > S_2_SNf + S_7_E
    k_unbind_n * T_2_7_SN
R447:
    S_2_NSf + S_7_E > T_2_7_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_7_E
R448:
    T_2_7_NS > S_2_NSf + S_7_E
    k_unbind_s * T_2_7_NS
R449:
    S_2_NNf + S_7_E > T_2_7_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_7_E
R450:
    T_2_7_NN > S_2_NNf + S_7_E
    k_unbind_n * T_2_7_NN
R451:
    S_2_S + S_8_N > T_2_8_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_8_N
R452:
    T_2_8_SN > S_2_S + S_8_N
    k_dissociate * T_2_8_SN
R453:
    S_2_N + S_8_S > T_2_8_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_8_S
R454:
    T_2_8_NS > S_2_N + S_8_S
    k_dissociate * T_2_8_NS
R455:
    S_2_N + S_8_N > T_2_8_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_8_N * 2.0
R456:
    T_2_8_NN > S_2_N + S_8_N
    k_dissociate * T_2_8_NN
R457:
    S_2_SNf + S_8_E > T_2_8_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_8_E
R458:
    T_2_8_SN > S_2_SNf + S_8_E
    k_unbind_n * T_2_8_SN
R459:
    S_2_NSf + S_8_E > T_2_8_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_8_E
R460:
    T_2_8_NS > S_2_NSf + S_8_E
    k_unbind_s * T_2_8_NS
R461:
    S_2_NNf + S_8_E > T_2_8_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_8_E
R462:
    T_2_8_NN > S_2_NNf + S_8_E
    k_unbind_n * T_2_8_NN
R463:
    S_2_S + S_9_N > T_2_9_SN
    k_dimerise * 0.1111111111111111 * S_2_S * S_9_N
R464:
    T_2_9_SN > S_2_S + S_9_N
    k_dissociate * T_2_9_SN
R465:
    S_2_N + S_9_S > T_2_9_NS
    k_dimerise * 0.1111111111111111 * S_2_N * S_9_S
R466:
    T_2_9_NS > S_2_N + S_9_S
    k_dissociate * T_2_9_NS
R467:
    S_2_N + S_9_N > T_2_9_NN
    k_dimerise * 0.1111111111111111 * S_2_N * S_9_N * 2.0
R468:
    T_2_9_NN > S_2_N + S_9_N
    k_dissociate * T_2_9_NN
R469:
    S_2_SNf + S_9_E > T_2_9_SN
    k_bind_n * 0.1111111111111111 * S_2_SNf * S_9_E
R470:
    T_2_9_SN > S_2_SNf + S_9_E
    k_unbind_n * T_2_9_SN
R471:
    S_2_NSf + S_9_E > T_2_9_NS
    k_bind_s * 0.1111111111111111 * S_2_NSf * S_9_E
R472:
    T_2_9_NS > S_2_NSf + S_9_E
    k_unbind_s * T_2_9_NS
R473:
    S_2_NNf + S_9_E > T_2_9_NN
    k_bind_n * 0.1111111111111111 * S_2_NNf * S_9_E
R474:
    T_2_9_NN > S_2_NNf + S_9_E
    k_unbind_n * T_2_9_NN
R475:
    S_3_SNf + S_0_E > T_0_3_NS
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_0_E
R476:
    T_0_3_NS > S_3_SNf + S_0_E
    k_unbind_n * T_0_3_NS
R477:
    S_3_NSf + S_0_E > T_0_3_SN
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_0_E
R478:
    T_0_3_SN > S_3_NSf + S_0_E
    k_unbind_s * T_0_3_SN
R479:
    S_3_NNf + S_0_E > T_0_3_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_0_E
R480:
    T_0_3_NN > S_3_NNf + S_0_E
    k_unbind_n * T_0_3_NN
R481:
    S_3_SNf + S_1_E > T_1_3_NS
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_1_E
R482:
    T_1_3_NS > S_3_SNf + S_1_E
    k_unbind_n * T_1_3_NS
R483:
    S_3_NSf + S_1_E > T_1_3_SN
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_1_E
R484:
    T_1_3_SN > S_3_NSf + S_1_E
    k_unbind_s * T_1_3_SN
R485:
    S_3_NNf + S_1_E > T_1_3_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_1_E
R486:
    T_1_3_NN > S_3_NNf + S_1_E
    k_unbind_n * T_1_3_NN
R487:
    S_3_SNf + S_2_E > T_2_3_NS
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_2_E
R488:
    T_2_3_NS > S_3_SNf + S_2_E
    k_unbind_n * T_2_3_NS
R489:
    S_3_NSf + S_2_E > T_2_3_SN
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_2_E
R490:
    T_2_3_SN > S_3_NSf + S_2_E
    k_unbind_s * T_2_3_SN
R491:
    S_3_NNf + S_2_E > T_2_3_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_2_E
R492:
    T_2_3_NN > S_3_NNf + S_2_E
    k_unbind_n * T_2_3_NN
R493:
    S_3_S + S_4_N > T_3_4_SN
    k_dimerise * 0.1111111111111111 * S_3_S * S_4_N
R494:
    T_3_4_SN > S_3_S + S_4_N
    k_dissociate * T_3_4_SN
R495:
    S_3_N + S_4_S > T_3_4_NS
    k_dimerise * 0.1111111111111111 * S_3_N * S_4_S
R496:
    T_3_4_NS > S_3_N + S_4_S
    k_dissociate * T_3_4_NS
R497:
    S_3_N + S_4_N > T_3_4_NN
    k_dimerise * 0.1111111111111111 * S_3_N * S_4_N * 2.0
R498:
    T_3_4_NN > S_3_N + S_4_N
    k_dissociate * T_3_4_NN
R499:
    S_3_SNf + S_4_E > T_3_4_SN
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_4_E
R500:
    T_3_4_SN > S_3_SNf + S_4_E
    k_unbind_n * T_3_4_SN
R501:
    S_3_NSf + S_4_E > T_3_4_NS
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_4_E
R502:
    T_3_4_NS > S_3_NSf + S_4_E
    k_unbind_s * T_3_4_NS
R503:
    S_3_NNf + S_4_E > T_3_4_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_4_E
R504:
    T_3_4_NN > S_3_NNf + S_4_E
    k_unbind_n * T_3_4_NN
R505:
    S_3_S + S_5_N > T_3_5_SN
    k_dimerise * 0.1111111111111111 * S_3_S * S_5_N
R506:
    T_3_5_SN > S_3_S + S_5_N
    k_dissociate * T_3_5_SN
R507:
    S_3_N + S_5_S > T_3_5_NS
    k_dimerise * 0.1111111111111111 * S_3_N * S_5_S
R508:
    T_3_5_NS > S_3_N + S_5_S
    k_dissociate * T_3_5_NS
R509:
    S_3_N + S_5_N > T_3_5_NN
    k_dimerise * 0.1111111111111111 * S_3_N * S_5_N * 2.0
R510:
    T_3_5_NN > S_3_N + S_5_N
    k_dissociate * T_3_5_NN
R511:
    S_3_SNf + S_5_E > T_3_5_SN
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_5_E
R512:
    T_3_5_SN > S_3_SNf + S_5_E
    k_unbind_n * T_3_5_SN
R513:
    S_3_NSf + S_5_E > T_3_5_NS
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_5_E
R514:
    T_3_5_NS > S_3_NSf + S_5_E
    k_unbind_s * T_3_5_NS
R515:
    S_3_NNf + S_5_E > T_3_5_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_5_E
R516:
    T_3_5_NN > S_3_NNf + S_5_E
    k_unbind_n * T_3_5_NN
R517:
    S_3_S + S_6_N > T_3_6_SN
    k_dimerise * 0.1111111111111111 * S_3_S * S_6_N
R518:
    T_3_6_SN > S_3_S + S_6_N
    k_dissociate * T_3_6_SN
R519:
    S_3_N + S_6_S > T_3_6_NS
    k_dimerise * 0.1111111111111111 * S_3_N * S_6_S
R520:
    T_3_6_NS > S_3_N + S_6_S
    k_dissociate * T_3_6_NS
R521:
    S_3_N + S_6_N > T_3_6_NN
    k_dimerise * 0.1111111111111111 * S_3_N * S_6_N * 2.0
R522:
    T_3_6_NN > S_3_N + S_6_N
    k_dissociate * T_3_6_NN
R523:
    S_3_SNf + S_6_E > T_3_6_SN
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_6_E
R524:
    T_3_6_SN > S_3_SNf + S_6_E
    k_unbind_n * T_3_6_SN
R525:
    S_3_NSf + S_6_E > T_3_6_NS
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_6_E
R526:
    T_3_6_NS > S_3_NSf + S_6_E
    k_unbind_s * T_3_6_NS
R527:
    S_3_NNf + S_6_E > T_3_6_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_6_E
R528:
    T_3_6_NN > S_3_NNf + S_6_E
    k_unbind_n * T_3_6_NN
R529:
    S_3_S + S_7_N > T_3_7_SN
    k_dimerise * 0.1111111111111111 * S_3_S * S_7_N
R530:
    T_3_7_SN > S_3_S + S_7_N
    k_dissociate * T_3_7_SN
R531:
    S_3_N + S_7_S > T_3_7_NS
    k_dimerise * 0.1111111111111111 * S_3_N * S_7_S
R532:
    T_3_7_NS > S_3_N + S_7_S
    k_dissociate * T_3_7_NS
R533:
    S_3_N + S_7_N > T_3_7_NN
    k_dimerise * 0.1111111111111111 * S_3_N * S_7_N * 2.0
R534:
    T_3_7_NN > S_3_N + S_7_N
    k_dissociate * T_3_7_NN
R535:
    S_3_SNf + S_7_E > T_3_7_SN
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_7_E
R536:
    T_3_7_SN > S_3_SNf + S_7_E
    k_unbind_n * T_3_7_SN
R537:
    S_3_NSf + S_7_E > T_3_7_NS
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_7_E
R538:
    T_3_7_NS > S_3_NSf + S_7_E
    k_unbind_s * T_3_7_NS
R539:
    S_3_NNf + S_7_E > T_3_7_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_7_E
R540:
    T_3_7_NN > S_3_NNf + S_7_E
    k_unbind_n * T_3_7_NN
R541:
    S_3_S + S_8_N > T_3_8_SN
    k_dimerise * 0.1111111111111111 * S_3_S * S_8_N
R542:
    T_3_8_SN > S_3_S + S_8_N
    k_dissociate * T_3_8_SN
R543:
    S_3_N + S_8_S > T_3_8_NS
    k_dimerise * 0.1111111111111111 * S_3_N * S_8_S
R544:
    T_3_8_NS > S_3_N + S_8_S
    k_dissociate * T_3_8_NS
R545:
    S_3_N + S_8_N > T_3_8_NN
    k_dimerise * 0.1111111111111111 * S_3_N * S_8_N * 2.0
R546:
    T_3_8_NN > S_3_N + S_8_N
    k_dissociate * T_3_8_NN
R547:
    S_3_SNf + S_8_E > T_3_8_SN
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_8_E
R548:
    T_3_8_SN > S_3_SNf + S_8_E
    k_unbind_n * T_3_8_SN
R549:
    S_3_NSf + S_8_E > T_3_8_NS
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_8_E
R550:
    T_3_8_NS > S_3_NSf + S_8_E
    k_unbind_s * T_3_8_NS
R551:
    S_3_NNf + S_8_E > T_3_8_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_8_E
R552:
    T_3_8_NN > S_3_NNf + S_8_E
    k_unbind_n * T_3_8_NN
R553:
    S_3_S + S_9_N > T_3_9_SN
    k_dimerise * 0.1111111111111111 * S_3_S * S_9_N
R554:
    T_3_9_SN > S_3_S + S_9_N
    k_dissociate * T_3_9_SN
R555:
    S_3_N + S_9_S > T_3_9_NS
    k_dimerise * 0.1111111111111111 * S_3_N * S_9_S
R556:
    T_3_9_NS > S_3_N + S_9_S
    k_dissociate * T_3_9_NS
R557:
    S_3_N + S_9_N > T_3_9_NN
    k_dimerise * 0.1111111111111111 * S_3_N * S_9_N * 2.0
R558:
    T_3_9_NN > S_3_N + S_9_N
    k_dissociate * T_3_9_NN
R559:
    S_3_SNf + S_9_E > T_3_9_SN
    k_bind_n * 0.1111111111111111 * S_3_SNf * S_9_E
R560:
    T_3_9_SN > S_3_SNf + S_9_E
    k_unbind_n * T_3_9_SN
R561:
    S_3_NSf + S_9_E > T_3_9_NS
    k_bind_s * 0.1111111111111111 * S_3_NSf * S_9_E
R562:
    T_3_9_NS > S_3_NSf + S_9_E
    k_unbind_s * T_3_9_NS
R563:
    S_3_NNf + S_9_E > T_3_9_NN
    k_bind_n * 0.1111111111111111 * S_3_NNf * S_9_E
R564:
    T_3_9_NN > S_3_NNf + S_9_E
    k_unbind_n * T_3_9_NN
R565:
    S_4_SNf + S_0_E > T_0_4_NS
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_0_E
R566:
    T_0_4_NS > S_4_SNf + S_0_E
    k_unbind_n * T_0_4_NS
R567:
    S_4_NSf + S_0_E > T_0_4_SN
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_0_E
R568:
    T_0_4_SN > S_4_NSf + S_0_E
    k_unbind_s * T_0_4_SN
R569:
    S_4_NNf + S_0_E > T_0_4_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_0_E
R570:
    T_0_4_NN > S_4_NNf + S_0_E
    k_unbind_n * T_0_4_NN
R571:
    S_4_SNf + S_1_E > T_1_4_NS
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_1_E
R572:
    T_1_4_NS > S_4_SNf + S_1_E
    k_unbind_n * T_1_4_NS
R573:
    S_4_NSf + S_1_E > T_1_4_SN
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_1_E
R574:
    T_1_4_SN > S_4_NSf + S_1_E
    k_unbind_s * T_1_4_SN
R575:
    S_4_NNf + S_1_E > T_1_4_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_1_E
R576:
    T_1_4_NN > S_4_NNf + S_1_E
    k_unbind_n * T_1_4_NN
R577:
    S_4_SNf + S_2_E > T_2_4_NS
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_2_E
R578:
    T_2_4_NS > S_4_SNf + S_2_E
    k_unbind_n * T_2_4_NS
R579:
    S_4_NSf + S_2_E > T_2_4_SN
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_2_E
R580:
    T_2_4_SN > S_4_NSf + S_2_E
    k_unbind_s * T_2_4_SN
R581:
    S_4_NNf + S_2_E > T_2_4_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_2_E
R582:
    T_2_4_NN > S_4_NNf + S_2_E
    k_unbind_n * T_2_4_NN
R583:
    S_4_SNf + S_3_E > T_3_4_NS
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_3_E
R584:
    T_3_4_NS > S_4_SNf + S_3_E
    k_unbind_n * T_3_4_NS
R585:
    S_4_NSf + S_3_E > T_3_4_SN
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_3_E
R586:
    T_3_4_SN > S_4_NSf + S_3_E
    k_unbind_s * T_3_4_SN
R587:
    S_4_NNf + S_3_E > T_3_4_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_3_E
R588:
    T_3_4_NN > S_4_NNf + S_3_E
    k_unbind_n * T_3_4_NN
R589:
    S_4_S + S_5_N > T_4_5_SN
    k_dimerise * 0.1111111111111111 * S_4_S * S_5_N
R590:
    T_4_5_SN > S_4_S + S_5_N
    k_dissociate * T_4_5_SN
R591:
    S_4_N + S_5_S > T_4_5_NS
    k_dimerise * 0.1111111111111111 * S_4_N * S_5_S
R592:
    T_4_5_NS > S_4_N + S_5_S
    k_dissociate * T_4_5_NS
R593:
    S_4_N + S_5_N > T_4_5_NN
    k_dimerise * 0.1111111111111111 * S_4_N * S_5_N * 2.0
R594:
    T_4_5_NN > S_4_N + S_5_N
    k_dissociate * T_4_5_NN
R595:
    S_4_SNf + S_5_E > T_4_5_SN
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_5_E
R596:
    T_4_5_SN > S_4_SNf + S_5_E
    k_unbind_n * T_4_5_SN
R597:
    S_4_NSf + S_5_E > T_4_5_NS
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_5_E
R598:
    T_4_5_NS > S_4_NSf + S_5_E
    k_unbind_s * T_4_5_NS
R599:
    S_4_NNf + S_5_E > T_4_5_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_5_E
R600:
    T_4_5_NN > S_4_NNf + S_5_E
    k_unbind_n * T_4_5_NN
R601:
    S_4_S + S_6_N > T_4_6_SN
    k_dimerise * 0.1111111111111111 * S_4_S * S_6_N
R602:
    T_4_6_SN > S_4_S + S_6_N
    k_dissociate * T_4_6_SN
R603:
    S_4_N + S_6_S > T_4_6_NS
    k_dimerise * 0.1111111111111111 * S_4_N * S_6_S
R604:
    T_4_6_NS > S_4_N + S_6_S
    k_dissociate * T_4_6_NS
R605:
    S_4_N + S_6_N > T_4_6_NN
    k_dimerise * 0.1111111111111111 * S_4_N * S_6_N * 2.0
R606:
    T_4_6_NN > S_4_N + S_6_N
    k_dissociate * T_4_6_NN
R607:
    S_4_SNf + S_6_E > T_4_6_SN
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_6_E
R608:
    T_4_6_SN > S_4_SNf + S_6_E
    k_unbind_n * T_4_6_SN
R609:
    S_4_NSf + S_6_E > T_4_6_NS
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_6_E
R610:
    T_4_6_NS > S_4_NSf + S_6_E
    k_unbind_s * T_4_6_NS
R611:
    S_4_NNf + S_6_E > T_4_6_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_6_E
R612:
    T_4_6_NN > S_4_NNf + S_6_E
    k_unbind_n * T_4_6_NN
R613:
    S_4_S + S_7_N > T_4_7_SN
    k_dimerise * 0.1111111111111111 * S_4_S * S_7_N
R614:
    T_4_7_SN > S_4_S + S_7_N
    k_dissociate * T_4_7_SN
R615:
    S_4_N + S_7_S > T_4_7_NS
    k_dimerise * 0.1111111111111111 * S_4_N * S_7_S
R616:
    T_4_7_NS > S_4_N + S_7_S
    k_dissociate * T_4_7_NS
R617:
    S_4_N + S_7_N > T_4_7_NN
    k_dimerise * 0.1111111111111111 * S_4_N * S_7_N * 2.0
R618:
    T_4_7_NN > S_4_N + S_7_N
    k_dissociate * T_4_7_NN
R619:
    S_4_SNf + S_7_E > T_4_7_SN
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_7_E
R620:
    T_4_7_SN > S_4_SNf + S_7_E
    k_unbind_n * T_4_7_SN
R621:
    S_4_NSf + S_7_E > T_4_7_NS
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_7_E
R622:
    T_4_7_NS > S_4_NSf + S_7_E
    k_unbind_s * T_4_7_NS
R623:
    S_4_NNf + S_7_E > T_4_7_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_7_E
R624:
    T_4_7_NN > S_4_NNf + S_7_E
    k_unbind_n * T_4_7_NN
R625:
    S_4_S + S_8_N > T_4_8_SN
    k_dimerise * 0.1111111111111111 * S_4_S * S_8_N
R626:
    T_4_8_SN > S_4_S + S_8_N
    k_dissociate * T_4_8_SN
R627:
    S_4_N + S_8_S > T_4_8_NS
    k_dimerise * 0.1111111111111111 * S_4_N * S_8_S
R628:
    T_4_8_NS > S_4_N + S_8_S
    k_dissociate * T_4_8_NS
R629:
    S_4_N + S_8_N > T_4_8_NN
    k_dimerise * 0.1111111111111111 * S_4_N * S_8_N * 2.0
R630:
    T_4_8_NN > S_4_N + S_8_N
    k_dissociate * T_4_8_NN
R631:
    S_4_SNf + S_8_E > T_4_8_SN
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_8_E
R632:
    T_4_8_SN > S_4_SNf + S_8_E
    k_unbind_n * T_4_8_SN
R633:
    S_4_NSf + S_8_E > T_4_8_NS
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_8_E
R634:
    T_4_8_NS > S_4_NSf + S_8_E
    k_unbind_s * T_4_8_NS
R635:
    S_4_NNf + S_8_E > T_4_8_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_8_E
R636:
    T_4_8_NN > S_4_NNf + S_8_E
    k_unbind_n * T_4_8_NN
R637:
    S_4_S + S_9_N > T_4_9_SN
    k_dimerise * 0.1111111111111111 * S_4_S * S_9_N
R638:
    T_4_9_SN > S_4_S + S_9_N
    k_dissociate * T_4_9_SN
R639:
    S_4_N + S_9_S > T_4_9_NS
    k_dimerise * 0.1111111111111111 * S_4_N * S_9_S
R640:
    T_4_9_NS > S_4_N + S_9_S
    k_dissociate * T_4_9_NS
R641:
    S_4_N + S_9_N > T_4_9_NN
    k_dimerise * 0.1111111111111111 * S_4_N * S_9_N * 2.0
R642:
    T_4_9_NN > S_4_N + S_9_N
    k_dissociate * T_4_9_NN
R643:
    S_4_SNf + S_9_E > T_4_9_SN
    k_bind_n * 0.1111111111111111 * S_4_SNf * S_9_E
R644:
    T_4_9_SN > S_4_SNf + S_9_E
    k_unbind_n * T_4_9_SN
R645:
    S_4_NSf + S_9_E > T_4_9_NS
    k_bind_s * 0.1111111111111111 * S_4_NSf * S_9_E
R646:
    T_4_9_NS > S_4_NSf + S_9_E
    k_unbind_s * T_4_9_NS
R647:
    S_4_NNf + S_9_E > T_4_9_NN
    k_bind_n * 0.1111111111111111 * S_4_NNf * S_9_E
R648:
    T_4_9_NN > S_4_NNf + S_9_E
    k_unbind_n * T_4_9_NN
R649:
    S_5_SNf + S_0_E > T_0_5_NS
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_0_E
R650:
    T_0_5_NS > S_5_SNf + S_0_E
    k_unbind_n * T_0_5_NS
R651:
    S_5_NSf + S_0_E > T_0_5_SN
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_0_E
R652:
    T_0_5_SN > S_5_NSf + S_0_E
    k_unbind_s * T_0_5_SN
R653:
    S_5_NNf + S_0_E > T_0_5_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_0_E
R654:
    T_0_5_NN > S_5_NNf + S_0_E
    k_unbind_n * T_0_5_NN
R655:
    S_5_SNf + S_1_E > T_1_5_NS
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_1_E
R656:
    T_1_5_NS > S_5_SNf + S_1_E
    k_unbind_n * T_1_5_NS
R657:
    S_5_NSf + S_1_E > T_1_5_SN
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_1_E
R658:
    T_1_5_SN > S_5_NSf + S_1_E
    k_unbind_s * T_1_5_SN
R659:
    S_5_NNf + S_1_E > T_1_5_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_1_E
R660:
    T_1_5_NN > S_5_NNf + S_1_E
    k_unbind_n * T_1_5_NN
R661:
    S_5_SNf + S_2_E > T_2_5_NS
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_2_E
R662:
    T_2_5_NS > S_5_SNf + S_2_E
    k_unbind_n * T_2_5_NS
R663:
    S_5_NSf + S_2_E > T_2_5_SN
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_2_E
R664:
    T_2_5_SN > S_5_NSf + S_2_E
    k_unbind_s * T_2_5_SN
R665:
    S_5_NNf + S_2_E > T_2_5_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_2_E
R666:
    T_2_5_NN > S_5_NNf + S_2_E
    k_unbind_n * T_2_5_NN
R667:
    S_5_SNf + S_3_E > T_3_5_NS
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_3_E
R668:
    T_3_5_NS > S_5_SNf + S_3_E
    k_unbind_n * T_3_5_NS
R669:
    S_5_NSf + S_3_E > T_3_5_SN
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_3_E
R670:
    T_3_5_SN > S_5_NSf + S_3_E
    k_unbind_s * T_3_5_SN
R671:
    S_5_NNf + S_3_E > T_3_5_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_3_E
R672:
    T_3_5_NN > S_5_NNf + S_3_E
    k_unbind_n * T_3_5_NN
R673:
    S_5_SNf + S_4_E > T_4_5_NS
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_4_E
R674:
    T_4_5_NS > S_5_SNf + S_4_E
    k_unbind_n * T_4_5_NS
R675:
    S_5_NSf + S_4_E > T_4_5_SN
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_4_E
R676:
    T_4_5_SN > S_5_NSf + S_4_E
    k_unbind_s * T_4_5_SN
R677:
    S_5_NNf + S_4_E > T_4_5_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_4_E
R678:
    T_4_5_NN > S_5_NNf + S_4_E
    k_unbind_n * T_4_5_NN
R679:
    S_5_S + S_6_N > T_5_6_SN
    k_dimerise * 0.1111111111111111 * S_5_S * S_6_N
R680:
    T_5_6_SN > S_5_S + S_6_N
    k_dissociate * T_5_6_SN
R681:
    S_5_N + S_6_S > T_5_6_NS
    k_dimerise * 0.1111111111111111 * S_5_N * S_6_S
R682:
    T_5_6_NS > S_5_N + S_6_S
    k_dissociate * T_5_6_NS
R683:
    S_5_N + S_6_N > T_5_6_NN
    k_dimerise * 0.1111111111111111 * S_5_N * S_6_N * 2.0
R684:
    T_5_6_NN > S_5_N + S_6_N
    k_dissociate * T_5_6_NN
R685:
    S_5_SNf + S_6_E > T_5_6_SN
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_6_E
R686:
    T_5_6_SN > S_5_SNf + S_6_E
    k_unbind_n * T_5_6_SN
R687:
    S_5_NSf + S_6_E > T_5_6_NS
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_6_E
R688:
    T_5_6_NS > S_5_NSf + S_6_E
    k_unbind_s * T_5_6_NS
R689:
    S_5_NNf + S_6_E > T_5_6_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_6_E
R690:
    T_5_6_NN > S_5_NNf + S_6_E
    k_unbind_n * T_5_6_NN
R691:
    S_5_S + S_7_N > T_5_7_SN
    k_dimerise * 0.1111111111111111 * S_5_S * S_7_N
R692:
    T_5_7_SN > S_5_S + S_7_N
    k_dissociate * T_5_7_SN
R693:
    S_5_N + S_7_S > T_5_7_NS
    k_dimerise * 0.1111111111111111 * S_5_N * S_7_S
R694:
    T_5_7_NS > S_5_N + S_7_S
    k_dissociate * T_5_7_NS
R695:
    S_5_N + S_7_N > T_5_7_NN
    k_dimerise * 0.1111111111111111 * S_5_N * S_7_N * 2.0
R696:
    T_5_7_NN > S_5_N + S_7_N
    k_dissociate * T_5_7_NN
R697:
    S_5_SNf + S_7_E > T_5_7_SN
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_7_E
R698:
    T_5_7_SN > S_5_SNf + S_7_E
    k_unbind_n * T_5_7_SN
R699:
    S_5_NSf + S_7_E > T_5_7_NS
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_7_E
R700:
    T_5_7_NS > S_5_NSf + S_7_E
    k_unbind_s * T_5_7_NS
R701:
    S_5_NNf + S_7_E > T_5_7_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_7_E
R702:
    T_5_7_NN > S_5_NNf + S_7_E
    k_unbind_n * T_5_7_NN
R703:
    S_5_S + S_8_N > T_5_8_SN
    k_dimerise * 0.1111111111111111 * S_5_S * S_8_N
R704:
    T_5_8_SN > S_5_S + S_8_N
    k_dissociate * T_5_8_SN
R705:
    S_5_N + S_8_S > T_5_8_NS
    k_dimerise * 0.1111111111111111 * S_5_N * S_8_S
R706:
    T_5_8_NS > S_5_N + S_8_S
    k_dissociate * T_5_8_NS
R707:
    S_5_N + S_8_N > T_5_8_NN
    k_dimerise * 0.1111111111111111 * S_5_N * S_8_N * 2.0
R708:
    T_5_8_NN > S_5_N + S_8_N
    k_dissociate * T_5_8_NN
R709:
    S_5_SNf + S_8_E > T_5_8_SN
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_8_E
R710:
    T_5_8_SN > S_5_SNf + S_8_E
    k_unbind_n * T_5_8_SN
R711:
    S_5_NSf + S_8_E > T_5_8_NS
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_8_E
R712:
    T_5_8_NS > S_5_NSf + S_8_E
    k_unbind_s * T_5_8_NS
R713:
    S_5_NNf + S_8_E > T_5_8_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_8_E
R714:
    T_5_8_NN > S_5_NNf + S_8_E
    k_unbind_n * T_5_8_NN
R715:
    S_5_S + S_9_N > T_5_9_SN
    k_dimerise * 0.1111111111111111 * S_5_S * S_9_N
R716:
    T_5_9_SN > S_5_S + S_9_N
    k_dissociate * T_5_9_SN
R717:
    S_5_N + S_9_S > T_5_9_NS
    k_dimerise * 0.1111111111111111 * S_5_N * S_9_S
R718:
    T_5_9_NS > S_5_N + S_9_S
    k_dissociate * T_5_9_NS
R719:
    S_5_N + S_9_N > T_5_9_NN
    k_dimerise * 0.1111111111111111 * S_5_N * S_9_N * 2.0
R720:
    T_5_9_NN > S_5_N + S_9_N
    k_dissociate * T_5_9_NN
R721:
    S_5_SNf + S_9_E > T_5_9_SN
    k_bind_n * 0.1111111111111111 * S_5_SNf * S_9_E
R722:
    T_5_9_SN > S_5_SNf + S_9_E
    k_unbind_n * T_5_9_SN
R723:
    S_5_NSf + S_9_E > T_5_9_NS
    k_bind_s * 0.1111111111111111 * S_5_NSf * S_9_E
R724:
    T_5_9_NS > S_5_NSf + S_9_E
    k_unbind_s * T_5_9_NS
R725:
    S_5_NNf + S_9_E > T_5_9_NN
    k_bind_n * 0.1111111111111111 * S_5_NNf * S_9_E
R726:
    T_5_9_NN > S_5_NNf + S_9_E
    k_unbind_n * T_5_9_NN
R727:
    S_6_SNf + S_0_E > T_0_6_NS
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_0_E
R728:
    T_0_6_NS > S_6_SNf + S_0_E
    k_unbind_n * T_0_6_NS
R729:
    S_6_NSf + S_0_E > T_0_6_SN
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_0_E
R730:
    T_0_6_SN > S_6_NSf + S_0_E
    k_unbind_s * T_0_6_SN
R731:
    S_6_NNf + S_0_E > T_0_6_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_0_E
R732:
    T_0_6_NN > S_6_NNf + S_0_E
    k_unbind_n * T_0_6_NN
R733:
    S_6_SNf + S_1_E > T_1_6_NS
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_1_E
R734:
    T_1_6_NS > S_6_SNf + S_1_E
    k_unbind_n * T_1_6_NS
R735:
    S_6_NSf + S_1_E > T_1_6_SN
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_1_E
R736:
    T_1_6_SN > S_6_NSf + S_1_E
    k_unbind_s * T_1_6_SN
R737:
    S_6_NNf + S_1_E > T_1_6_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_1_E
R738:
    T_1_6_NN > S_6_NNf + S_1_E
    k_unbind_n * T_1_6_NN
R739:
    S_6_SNf + S_2_E > T_2_6_NS
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_2_E
R740:
    T_2_6_NS > S_6_SNf + S_2_E
    k_unbind_n * T_2_6_NS
R741:
    S_6_NSf + S_2_E > T_2_6_SN
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_2_E
R742:
    T_2_6_SN > S_6_NSf + S_2_E
    k_unbind_s * T_2_6_SN
R743:
    S_6_NNf + S_2_E > T_2_6_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_2_E
R744:
    T_2_6_NN > S_6_NNf + S_2_E
    k_unbind_n * T_2_6_NN
R745:
    S_6_SNf + S_3_E > T_3_6_NS
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_3_E
R746:
    T_3_6_NS > S_6_SNf + S_3_E
    k_unbind_n * T_3_6_NS
R747:
    S_6_NSf + S_3_E > T_3_6_SN
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_3_E
R748:
    T_3_6_SN > S_6_NSf + S_3_E
    k_unbind_s * T_3_6_SN
R749:
    S_6_NNf + S_3_E > T_3_6_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_3_E
R750:
    T_3_6_NN > S_6_NNf + S_3_E
    k_unbind_n * T_3_6_NN
R751:
    S_6_SNf + S_4_E > T_4_6_NS
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_4_E
R752:
    T_4_6_NS > S_6_SNf + S_4_E
    k_unbind_n * T_4_6_NS
R753:
    S_6_NSf + S_4_E > T_4_6_SN
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_4_E
R754:
    T_4_6_SN > S_6_NSf + S_4_E
    k_unbind_s * T_4_6_SN
R755:
    S_6_NNf + S_4_E > T_4_6_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_4_E
R756:
    T_4_6_NN > S_6_NNf + S_4_E
    k_unbind_n * T_4_6_NN
R757:
    S_6_SNf + S_5_E > T_5_6_NS
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_5_E
R758:
    T_5_6_NS > S_6_SNf + S_5_E
    k_unbind_n * T_5_6_NS
R759:
    S_6_NSf + S_5_E > T_5_6_SN
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_5_E
R760:
    T_5_6_SN > S_6_NSf + S_5_E
    k_unbind_s * T_5_6_SN
R761:
    S_6_NNf + S_5_E > T_5_6_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_5_E
R762:
    T_5_6_NN > S_6_NNf + S_5_E
    k_unbind_n * T_5_6_NN
R763:
    S_6_S + S_7_N > T_6_7_SN
    k_dimerise * 0.1111111111111111 * S_6_S * S_7_N
R764:
    T_6_7_SN > S_6_S + S_7_N
    k_dissociate * T_6_7_SN
R765:
    S_6_N + S_7_S > T_6_7_NS
    k_dimerise * 0.1111111111111111 * S_6_N * S_7_S
R766:
    T_6_7_NS > S_6_N + S_7_S
    k_dissociate * T_6_7_NS
R767:
    S_6_N + S_7_N > T_6_7_NN
    k_dimerise * 0.1111111111111111 * S_6_N * S_7_N * 2.0
R768:
    T_6_7_NN > S_6_N + S_7_N
    k_dissociate * T_6_7_NN
R769:
    S_6_SNf + S_7_E > T_6_7_SN
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_7_E
R770:
    T_6_7_SN > S_6_SNf + S_7_E
    k_unbind_n * T_6_7_SN
R771:
    S_6_NSf + S_7_E > T_6_7_NS
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_7_E
R772:
    T_6_7_NS > S_6_NSf + S_7_E
    k_unbind_s * T_6_7_NS
R773:
    S_6_NNf + S_7_E > T_6_7_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_7_E
R774:
    T_6_7_NN > S_6_NNf + S_7_E
    k_unbind_n * T_6_7_NN
R775:
    S_6_S + S_8_N > T_6_8_SN
    k_dimerise * 0.1111111111111111 * S_6_S * S_8_N
R776:
    T_6_8_SN > S_6_S + S_8_N
    k_dissociate * T_6_8_SN
R777:
    S_6_N + S_8_S > T_6_8_NS
    k_dimerise * 0.1111111111111111 * S_6_N * S_8_S
R778:
    T_6_8_NS > S_6_N + S_8_S
    k_dissociate * T_6_8_NS
R779:
    S_6_N + S_8_N > T_6_8_NN
    k_dimerise * 0.1111111111111111 * S_6_N * S_8_N * 2.0
R780:
    T_6_8_NN > S_6_N + S_8_N
    k_dissociate * T_6_8_NN
R781:
    S_6_SNf + S_8_E > T_6_8_SN
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_8_E
R782:
    T_6_8_SN > S_6_SNf + S_8_E
    k_unbind_n * T_6_8_SN
R783:
    S_6_NSf + S_8_E > T_6_8_NS
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_8_E
R784:
    T_6_8_NS > S_6_NSf + S_8_E
    k_unbind_s * T_6_8_NS
R785:
    S_6_NNf + S_8_E > T_6_8_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_8_E
R786:
    T_6_8_NN > S_6_NNf + S_8_E
    k_unbind_n * T_6_8_NN
R787:
    S_6_S + S_9_N > T_6_9_SN
    k_dimerise * 0.1111111111111111 * S_6_S * S_9_N
R788:
    T_6_9_SN > S_6_S + S_9_N
    k_dissociate * T_6_9_SN
R789:
    S_6_N + S_9_S > T_6_9_NS
    k_dimerise * 0.1111111111111111 * S_6_N * S_9_S
R790:
    T_6_9_NS > S_6_N + S_9_S
    k_dissociate * T_6_9_NS
R791:
    S_6_N + S_9_N > T_6_9_NN
    k_dimerise * 0.1111111111111111 * S_6_N * S_9_N * 2.0
R792:
    T_6_9_NN > S_6_N + S_9_N
    k_dissociate * T_6_9_NN
R793:
    S_6_SNf + S_9_E > T_6_9_SN
    k_bind_n * 0.1111111111111111 * S_6_SNf * S_9_E
R794:
    T_6_9_SN > S_6_SNf + S_9_E
    k_unbind_n * T_6_9_SN
R795:
    S_6_NSf + S_9_E > T_6_9_NS
    k_bind_s * 0.1111111111111111 * S_6_NSf * S_9_E
R796:
    T_6_9_NS > S_6_NSf + S_9_E
    k_unbind_s * T_6_9_NS
R797:
    S_6_NNf + S_9_E > T_6_9_NN
    k_bind_n * 0.1111111111111111 * S_6_NNf * S_9_E
R798:
    T_6_9_NN > S_6_NNf + S_9_E
    k_unbind_n * T_6_9_NN
R799:
    S_7_SNf + S_0_E > T_0_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_0_E
R800:
    T_0_7_NS > S_7_SNf + S_0_E
    k_unbind_n * T_0_7_NS
R801:
    S_7_NSf + S_0_E > T_0_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_0_E
R802:
    T_0_7_SN > S_7_NSf + S_0_E
    k_unbind_s * T_0_7_SN
R803:
    S_7_NNf + S_0_E > T_0_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_0_E
R804:
    T_0_7_NN > S_7_NNf + S_0_E
    k_unbind_n * T_0_7_NN
R805:
    S_7_SNf + S_1_E > T_1_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_1_E
R806:
    T_1_7_NS > S_7_SNf + S_1_E
    k_unbind_n * T_1_7_NS
R807:
    S_7_NSf + S_1_E > T_1_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_1_E
R808:
    T_1_7_SN > S_7_NSf + S_1_E
    k_unbind_s * T_1_7_SN
R809:
    S_7_NNf + S_1_E > T_1_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_1_E
R810:
    T_1_7_NN > S_7_NNf + S_1_E
    k_unbind_n * T_1_7_NN
R811:
    S_7_SNf + S_2_E > T_2_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_2_E
R812:
    T_2_7_NS > S_7_SNf + S_2_E
    k_unbind_n * T_2_7_NS
R813:
    S_7_NSf + S_2_E > T_2_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_2_E
R814:
    T_2_7_SN > S_7_NSf + S_2_E
    k_unbind_s * T_2_7_SN
R815:
    S_7_NNf + S_2_E > T_2_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_2_E
R816:
    T_2_7_NN > S_7_NNf + S_2_E
    k_unbind_n * T_2_7_NN
R817:
    S_7_SNf + S_3_E > T_3_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_3_E
R818:
    T_3_7_NS > S_7_SNf + S_3_E
    k_unbind_n * T_3_7_NS
R819:
    S_7_NSf + S_3_E > T_3_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_3_E
R820:
    T_3_7_SN > S_7_NSf + S_3_E
    k_unbind_s * T_3_7_SN
R821:
    S_7_NNf + S_3_E > T_3_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_3_E
R822:
    T_3_7_NN > S_7_NNf + S_3_E
    k_unbind_n * T_3_7_NN
R823:
    S_7_SNf + S_4_E > T_4_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_4_E
R824:
    T_4_7_NS > S_7_SNf + S_4_E
    k_unbind_n * T_4_7_NS
R825:
    S_7_NSf + S_4_E > T_4_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_4_E
R826:
    T_4_7_SN > S_7_NSf + S_4_E
    k_unbind_s * T_4_7_SN
R827:
    S_7_NNf + S_4_E > T_4_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_4_E
R828:
    T_4_7_NN > S_7_NNf + S_4_E
    k_unbind_n * T_4_7_NN
R829:
    S_7_SNf + S_5_E > T_5_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_5_E
R830:
    T_5_7_NS > S_7_SNf + S_5_E
    k_unbind_n * T_5_7_NS
R831:
    S_7_NSf + S_5_E > T_5_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_5_E
R832:
    T_5_7_SN > S_7_NSf + S_5_E
    k_unbind_s * T_5_7_SN
R833:
    S_7_NNf + S_5_E > T_5_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_5_E
R834:
    T_5_7_NN > S_7_NNf + S_5_E
    k_unbind_n * T_5_7_NN
R835:
    S_7_SNf + S_6_E > T_6_7_NS
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_6_E
R836:
    T_6_7_NS > S_7_SNf + S_6_E
    k_unbind_n * T_6_7_NS
R837:
    S_7_NSf + S_6_E > T_6_7_SN
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_6_E
R838:
    T_6_7_SN > S_7_NSf + S_6_E
    k_unbind_s * T_6_7_SN
R839:
    S_7_NNf + S_6_E > T_6_7_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_6_E
R840:
    T_6_7_NN > S_7_NNf + S_6_E
    k_unbind_n * T_6_7_NN
R841:
    S_7_S + S_8_N > T_7_8_SN
    k_dimerise * 0.1111111111111111 * S_7_S * S_8_N
R842:
    T_7_8_SN > S_7_S + S_8_N
    k_dissociate * T_7_8_SN
R843:
    S_7_N + S_8_S > T_7_8_NS
    k_dimerise * 0.1111111111111111 * S_7_N * S_8_S
R844:
    T_7_8_NS > S_7_N + S_8_S
    k_dissociate * T_7_8_NS
R845:
    S_7_N + S_8_N > T_7_8_NN
    k_dimerise * 0.1111111111111111 * S_7_N * S_8_N * 2.0
R846:
    T_7_8_NN > S_7_N + S_8_N
    k_dissociate * T_7_8_NN
R847:
    S_7_SNf + S_8_E > T_7_8_SN
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_8_E
R848:
    T_7_8_SN > S_7_SNf + S_8_E
    k_unbind_n * T_7_8_SN
R849:
    S_7_NSf + S_8_E > T_7_8_NS
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_8_E
R850:
    T_7_8_NS > S_7_NSf + S_8_E
    k_unbind_s * T_7_8_NS
R851:
    S_7_NNf + S_8_E > T_7_8_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_8_E
R852:
    T_7_8_NN > S_7_NNf + S_8_E
    k_unbind_n * T_7_8_NN
R853:
    S_7_S + S_9_N > T_7_9_SN
    k_dimerise * 0.1111111111111111 * S_7_S * S_9_N
R854:
    T_7_9_SN > S_7_S + S_9_N
    k_dissociate * T_7_9_SN
R855:
    S_7_N + S_9_S > T_7_9_NS
    k_dimerise * 0.1111111111111111 * S_7_N * S_9_S
R856:
    T_7_9_NS > S_7_N + S_9_S
    k_dissociate * T_7_9_NS
R857:
    S_7_N + S_9_N > T_7_9_NN
    k_dimerise * 0.1111111111111111 * S_7_N * S_9_N * 2.0
R858:
    T_7_9_NN > S_7_N + S_9_N
    k_dissociate * T_7_9_NN
R859:
    S_7_SNf + S_9_E > T_7_9_SN
    k_bind_n * 0.1111111111111111 * S_7_SNf * S_9_E
R860:
    T_7_9_SN > S_7_SNf + S_9_E
    k_unbind_n * T_7_9_SN
R861:
    S_7_NSf + S_9_E > T_7_9_NS
    k_bind_s * 0.1111111111111111 * S_7_NSf * S_9_E
R862:
    T_7_9_NS > S_7_NSf + S_9_E
    k_unbind_s * T_7_9_NS
R863:
    S_7_NNf + S_9_E > T_7_9_NN
    k_bind_n * 0.1111111111111111 * S_7_NNf * S_9_E
R864:
    T_7_9_NN > S_7_NNf + S_9_E
    k_unbind_n * T_7_9_NN
R865:
    S_8_SNf + S_0_E > T_0_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_0_E
R866:
    T_0_8_NS > S_8_SNf + S_0_E
    k_unbind_n * T_0_8_NS
R867:
    S_8_NSf + S_0_E > T_0_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_0_E
R868:
    T_0_8_SN > S_8_NSf + S_0_E
    k_unbind_s * T_0_8_SN
R869:
    S_8_NNf + S_0_E > T_0_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_0_E
R870:
    T_0_8_NN > S_8_NNf + S_0_E
    k_unbind_n * T_0_8_NN
R871:
    S_8_SNf + S_1_E > T_1_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_1_E
R872:
    T_1_8_NS > S_8_SNf + S_1_E
    k_unbind_n * T_1_8_NS
R873:
    S_8_NSf + S_1_E > T_1_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_1_E
R874:
    T_1_8_SN > S_8_NSf + S_1_E
    k_unbind_s * T_1_8_SN
R875:
    S_8_NNf + S_1_E > T_1_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_1_E
R876:
    T_1_8_NN > S_8_NNf + S_1_E
    k_unbind_n * T_1_8_NN
R877:
    S_8_SNf + S_2_E > T_2_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_2_E
R878:
    T_2_8_NS > S_8_SNf + S_2_E
    k_unbind_n * T_2_8_NS
R879:
    S_8_NSf + S_2_E > T_2_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_2_E
R880:
    T_2_8_SN > S_8_NSf + S_2_E
    k_unbind_s * T_2_8_SN
R881:
    S_8_NNf + S_2_E > T_2_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_2_E
R882:
    T_2_8_NN > S_8_NNf + S_2_E
    k_unbind_n * T_2_8_NN
R883:
    S_8_SNf + S_3_E > T_3_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_3_E
R884:
    T_3_8_NS > S_8_SNf + S_3_E
    k_unbind_n * T_3_8_NS
R885:
    S_8_NSf + S_3_E > T_3_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_3_E
R886:
    T_3_8_SN > S_8_NSf + S_3_E
    k_unbind_s * T_3_8_SN
R887:
    S_8_NNf + S_3_E > T_3_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_3_E
R888:
    T_3_8_NN > S_8_NNf + S_3_E
    k_unbind_n * T_3_8_NN
R889:
    S_8_SNf + S_4_E > T_4_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_4_E
R890:
    T_4_8_NS > S_8_SNf + S_4_E
    k_unbind_n * T_4_8_NS
R891:
    S_8_NSf + S_4_E > T_4_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_4_E
R892:
    T_4_8_SN > S_8_NSf + S_4_E
    k_unbind_s * T_4_8_SN
R893:
    S_8_NNf + S_4_E > T_4_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_4_E
R894:
    T_4_8_NN > S_8_NNf + S_4_E
    k_unbind_n * T_4_8_NN
R895:
    S_8_SNf + S_5_E > T_5_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_5_E
R896:
    T_5_8_NS > S_8_SNf + S_5_E
    k_unbind_n * T_5_8_NS
R897:
    S_8_NSf + S_5_E > T_5_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_5_E
R898:
    T_5_8_SN > S_8_NSf + S_5_E
    k_unbind_s * T_5_8_SN
R899:
    S_8_NNf + S_5_E > T_5_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_5_E
R900:
    T_5_8_NN > S_8_NNf + S_5_E
    k_unbind_n * T_5_8_NN
R901:
    S_8_SNf + S_6_E > T_6_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_6_E
R902:
    T_6_8_NS > S_8_SNf + S_6_E
    k_unbind_n * T_6_8_NS
R903:
    S_8_NSf + S_6_E > T_6_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_6_E
R904:
    T_6_8_SN > S_8_NSf + S_6_E
    k_unbind_s * T_6_8_SN
R905:
    S_8_NNf + S_6_E > T_6_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_6_E
R906:
    T_6_8_NN > S_8_NNf + S_6_E
    k_unbind_n * T_6_8_NN
R907:
    S_8_SNf + S_7_E > T_7_8_NS
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_7_E
R908:
    T_7_8_NS > S_8_SNf + S_7_E
    k_unbind_n * T_7_8_NS
R909:
    S_8_NSf + S_7_E > T_7_8_SN
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_7_E
R910:
    T_7_8_SN > S_8_NSf + S_7_E
    k_unbind_s * T_7_8_SN
R911:
    S_8_NNf + S_7_E > T_7_8_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_7_E
R912:
    T_7_8_NN > S_8_NNf + S_7_E
    k_unbind_n * T_7_8_NN
R913:
    S_8_S + S_9_N > T_8_9_SN
    k_dimerise * 0.1111111111111111 * S_8_S * S_9_N
R914:
    T_8_9_SN > S_8_S + S_9_N
    k_dissociate * T_8_9_SN
R915:
    S_8_N + S_9_S > T_8_9_NS
    k_dimerise * 0.1111111111111111 * S_8_N * S_9_S
R916:
    T_8_9_NS > S_8_N + S_9_S
    k_dissociate * T_8_9_NS
R917:
    S_8_N + S_9_N > T_8_9_NN
    k_dimerise * 0.1111111111111111 * S_8_N * S_9_N * 2.0
R918:
    T_8_9_NN > S_8_N + S_9_N
    k_dissociate * T_8_9_NN
R919:
    S_8_SNf + S_9_E > T_8_9_SN
    k_bind_n * 0.1111111111111111 * S_8_SNf * S_9_E
R920:
    T_8_9_SN > S_8_SNf + S_9_E
    k_unbind_n * T_8_9_SN
R921:
    S_8_NSf + S_9_E > T_8_9_NS
    k_bind_s * 0.1111111111111111 * S_8_NSf * S_9_E
R922:
    T_8_9_NS > S_8_NSf + S_9_E
    k_unbind_s * T_8_9_NS
R923:
    S_8_NNf + S_9_E > T_8_9_NN
    k_bind_n * 0.1111111111111111 * S_8_NNf * S_9_E
R924:
    T_8_9_NN > S_8_NNf + S_9_E
    k_unbind_n * T_8_9_NN
R925:
    S_9_SNf + S_0_E > T_0_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_0_E
R926:
    T_0_9_NS > S_9_SNf + S_0_E
    k_unbind_n * T_0_9_NS
R927:
    S_9_NSf + S_0_E > T_0_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_0_E
R928:
    T_0_9_SN > S_9_NSf + S_0_E
    k_unbind_s * T_0_9_SN
R929:
    S_9_NNf + S_0_E > T_0_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_0_E
R930:
    T_0_9_NN > S_9_NNf + S_0_E
    k_unbind_n * T_0_9_NN
R931:
    S_9_SNf + S_1_E > T_1_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_1_E
R932:
    T_1_9_NS > S_9_SNf + S_1_E
    k_unbind_n * T_1_9_NS
R933:
    S_9_NSf + S_1_E > T_1_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_1_E
R934:
    T_1_9_SN > S_9_NSf + S_1_E
    k_unbind_s * T_1_9_SN
R935:
    S_9_NNf + S_1_E > T_1_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_1_E
R936:
    T_1_9_NN > S_9_NNf + S_1_E
    k_unbind_n * T_1_9_NN
R937:
    S_9_SNf + S_2_E > T_2_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_2_E
R938:
    T_2_9_NS > S_9_SNf + S_2_E
    k_unbind_n * T_2_9_NS
R939:
    S_9_NSf + S_2_E > T_2_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_2_E
R940:
    T_2_9_SN > S_9_NSf + S_2_E
    k_unbind_s * T_2_9_SN
R941:
    S_9_NNf + S_2_E > T_2_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_2_E
R942:
    T_2_9_NN > S_9_NNf + S_2_E
    k_unbind_n * T_2_9_NN
R943:
    S_9_SNf + S_3_E > T_3_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_3_E
R944:
    T_3_9_NS > S_9_SNf + S_3_E
    k_unbind_n * T_3_9_NS
R945:
    S_9_NSf + S_3_E > T_3_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_3_E
R946:
    T_3_9_SN > S_9_NSf + S_3_E
    k_unbind_s * T_3_9_SN
R947:
    S_9_NNf + S_3_E > T_3_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_3_E
R948:
    T_3_9_NN > S_9_NNf + S_3_E
    k_unbind_n * T_3_9_NN
R949:
    S_9_SNf + S_4_E > T_4_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_4_E
R950:
    T_4_9_NS > S_9_SNf + S_4_E
    k_unbind_n * T_4_9_NS
R951:
    S_9_NSf + S_4_E > T_4_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_4_E
R952:
    T_4_9_SN > S_9_NSf + S_4_E
    k_unbind_s * T_4_9_SN
R953:
    S_9_NNf + S_4_E > T_4_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_4_E
R954:
    T_4_9_NN > S_9_NNf + S_4_E
    k_unbind_n * T_4_9_NN
R955:
    S_9_SNf + S_5_E > T_5_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_5_E
R956:
    T_5_9_NS > S_9_SNf + S_5_E
    k_unbind_n * T_5_9_NS
R957:
    S_9_NSf + S_5_E > T_5_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_5_E
R958:
    T_5_9_SN > S_9_NSf + S_5_E
    k_unbind_s * T_5_9_SN
R959:
    S_9_NNf + S_5_E > T_5_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_5_E
R960:
    T_5_9_NN > S_9_NNf + S_5_E
    k_unbind_n * T_5_9_NN
R961:
    S_9_SNf + S_6_E > T_6_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_6_E
R962:
    T_6_9_NS > S_9_SNf + S_6_E
    k_unbind_n * T_6_9_NS
R963:
    S_9_NSf + S_6_E > T_6_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_6_E
R964:
    T_6_9_SN > S_9_NSf + S_6_E
    k_unbind_s * T_6_9_SN
R965:
    S_9_NNf + S_6_E > T_6_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_6_E
R966:
    T_6_9_NN > S_9_NNf + S_6_E
    k_unbind_n * T_6_9_NN
R967:
    S_9_SNf + S_7_E > T_7_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_7_E
R968:
    T_7_9_NS > S_9_SNf + S_7_E
    k_unbind_n * T_7_9_NS
R969:
    S_9_NSf + S_7_E > T_7_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_7_E
R970:
    T_7_9_SN > S_9_NSf + S_7_E
    k_unbind_s * T_7_9_SN
R971:
    S_9_NNf + S_7_E > T_7_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_7_E
R972:
    T_7_9_NN > S_9_NNf + S_7_E
    k_unbind_n * T_7_9_NN
R973:
    S_9_SNf + S_8_E > T_8_9_NS
    k_bind_n * 0.1111111111111111 * S_9_SNf * S_8_E
R974:
    T_8_9_NS > S_9_SNf + S_8_E
    k_unbind_n * T_8_9_NS
R975:
    S_9_NSf + S_8_E > T_8_9_SN
    k_bind_s * 0.1111111111111111 * S_9_NSf * S_8_E
R976:
    T_8_9_SN > S_9_NSf + S_8_E
    k_unbind_s * T_8_9_SN
R977:
    S_9_NNf + S_8_E > T_8_9_NN
    k_bind_n * 0.1111111111111111 * S_9_NNf * S_8_E
R978:
    T_8_9_NN > S_9_NNf + S_8_E
    k_unbind_n * T_8_9_NN
R979:
    $pool > mRNA
    k_prod_m * (S_4_S + S_4_N + S_4_SNf + S_4_NSf + S_4_NNf + T_0_4_SN + T_0_4_NS + T_0_4_NN + T_1_4_SN + T_1_4_NS + T_1_4_NN + T_2_4_SN + T_2_4_NS + T_2_4_NN + T_3_4_SN + T_3_4_NS + T_3_4_NN + T_4_5_SN + T_4_5_NS + T_4_5_NN + T_4_6_SN + T_4_6_NS + T_4_6_NN + T_4_7_SN + T_4_7_NS + T_4_7_NN + T_4_8_SN + T_4_8_NS + T_4_8_NN + T_4_9_SN + T_4_9_NS + T_4_9_NN)
R980:
    mRNA > $pool
    k_deg_m * mRNA

# --- Initial Conditions ---
S_free = 0
N_free = 0
SN_free = 0
NN_free = 0
mRNA = 0
S_0_E = 1
S_0_S = 0
S_0_N = 0
S_0_SNf = 0
S_0_NSf = 0
S_0_NNf = 0
S_1_E = 1
S_1_S = 0
S_1_N = 0
S_1_SNf = 0
S_1_NSf = 0
S_1_NNf = 0
S_2_E = 1
S_2_S = 0
S_2_N = 0
S_2_SNf = 0
S_2_NSf = 0
S_2_NNf = 0
S_3_E = 1
S_3_S = 0
S_3_N = 0
S_3_SNf = 0
S_3_NSf = 0
S_3_NNf = 0
S_4_E = 1
S_4_S = 0
S_4_N = 0
S_4_SNf = 0
S_4_NSf = 0
S_4_NNf = 0
S_5_E = 1
S_5_S = 0
S_5_N = 0
S_5_SNf = 0
S_5_NSf = 0
S_5_NNf = 0
S_6_E = 1
S_6_S = 0
S_6_N = 0
S_6_SNf = 0
S_6_NSf = 0
S_6_NNf = 0
S_7_E = 1
S_7_S = 0
S_7_N = 0
S_7_SNf = 0
S_7_NSf = 0
S_7_NNf = 0
S_8_E = 1
S_8_S = 0
S_8_N = 0
S_8_SNf = 0
S_8_NSf = 0
S_8_NNf = 0
S_9_E = 1
S_9_S = 0
S_9_N = 0
S_9_SNf = 0
S_9_NSf = 0
S_9_NNf = 0
T_0_1_SN = 0
T_0_1_NS = 0
T_0_1_NN = 0
T_0_2_SN = 0
T_0_2_NS = 0
T_0_2_NN = 0
T_0_3_SN = 0
T_0_3_NS = 0
T_0_3_NN = 0
T_0_4_SN = 0
T_0_4_NS = 0
T_0_4_NN = 0
T_0_5_SN = 0
T_0_5_NS = 0
T_0_5_NN = 0
T_0_6_SN = 0
T_0_6_NS = 0
T_0_6_NN = 0
T_0_7_SN = 0
T_0_7_NS = 0
T_0_7_NN = 0
T_0_8_SN = 0
T_0_8_NS = 0
T_0_8_NN = 0
T_0_9_SN = 0
T_0_9_NS = 0
T_0_9_NN = 0
T_1_2_SN = 0
T_1_2_NS = 0
T_1_2_NN = 0
T_1_3_SN = 0
T_1_3_NS = 0
T_1_3_NN = 0
T_1_4_SN = 0
T_1_4_NS = 0
T_1_4_NN = 0
T_1_5_SN = 0
T_1_5_NS = 0
T_1_5_NN = 0
T_1_6_SN = 0
T_1_6_NS = 0
T_1_6_NN = 0
T_1_7_SN = 0
T_1_7_NS = 0
T_1_7_NN = 0
T_1_8_SN = 0
T_1_8_NS = 0
T_1_8_NN = 0
T_1_9_SN = 0
T_1_9_NS = 0
T_1_9_NN = 0
T_2_3_SN = 0
T_2_3_NS = 0
T_2_3_NN = 0
T_2_4_SN = 0
T_2_4_NS = 0
T_2_4_NN = 0
T_2_5_SN = 0
T_2_5_NS = 0
T_2_5_NN = 0
T_2_6_SN = 0
T_2_6_NS = 0
T_2_6_NN = 0
T_2_7_SN = 0
T_2_7_NS = 0
T_2_7_NN = 0
T_2_8_SN = 0
T_2_8_NS = 0
T_2_8_NN = 0
T_2_9_SN = 0
T_2_9_NS = 0
T_2_9_NN = 0
T_3_4_SN = 0
T_3_4_NS = 0
T_3_4_NN = 0
T_3_5_SN = 0
T_3_5_NS = 0
T_3_5_NN = 0
T_3_6_SN = 0
T_3_6_NS = 0
T_3_6_NN = 0
T_3_7_SN = 0
T_3_7_NS = 0
T_3_7_NN = 0
T_3_8_SN = 0
T_3_8_NS = 0
T_3_8_NN = 0
T_3_9_SN = 0
T_3_9_NS = 0
T_3_9_NN = 0
T_4_5_SN = 0
T_4_5_NS = 0
T_4_5_NN = 0
T_4_6_SN = 0
T_4_6_NS = 0
T_4_6_NN = 0
T_4_7_SN = 0
T_4_7_NS = 0
T_4_7_NN = 0
T_4_8_SN = 0
T_4_8_NS = 0
T_4_8_NN = 0
T_4_9_SN = 0
T_4_9_NS = 0
T_4_9_NN = 0
T_5_6_SN = 0
T_5_6_NS = 0
T_5_6_NN = 0
T_5_7_SN = 0
T_5_7_NS = 0
T_5_7_NN = 0
T_5_8_SN = 0
T_5_8_NS = 0
T_5_8_NN = 0
T_5_9_SN = 0
T_5_9_NS = 0
T_5_9_NN = 0
T_6_7_SN = 0
T_6_7_NS = 0
T_6_7_NN = 0
T_6_8_SN = 0
T_6_8_NS = 0
T_6_8_NN = 0
T_6_9_SN = 0
T_6_9_NS = 0
T_6_9_NN = 0
T_7_8_SN = 0
T_7_8_NS = 0
T_7_8_NN = 0
T_7_9_SN = 0
T_7_9_NS = 0
T_7_9_NN = 0
T_8_9_SN = 0
T_8_9_NS = 0
T_8_9_NN = 0

# --- Parameters ---
k_s_in = 0.0
k_n_in = 0.0
k_s_out = 0.0
k_n_out = 0.0
k_bind_s = 0.0
k_unbind_s = 0.0
k_bind_n = 0.0
k_unbind_n = 0.0
k_dimerise = 0.0
k_dissociate = 0.0
k_prod_m = 0.0
k_deg_m = 0.0
