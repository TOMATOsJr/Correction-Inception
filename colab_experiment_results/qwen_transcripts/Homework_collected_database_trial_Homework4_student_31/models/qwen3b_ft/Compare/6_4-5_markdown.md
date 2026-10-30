![Source Image](6_4-5_source.png)
 

KCL @ \( N_1 \) node:

\[
\frac{V_1 - V_a}{R_1} + \frac{N_1 - V_2}{R_7} = 0
\]

\(\Rightarrow\) \( V_1 - V_a = -(V_1 - V_2) \frac{R_1}{R_7} \). \( R_1 \Rightarrow N_1 = V_1 + \frac{(V_1 - V_2) R_1}{R_7} = (1 + \frac{R_1}{R_7}) V_1 - \frac{R_1}{R_7} V_2 \)

\(\Rightarrow\) KCL @ \( N_2 \) node:

\[
\frac{V_2 - V_b}{R_2} - \frac{V_1 - V_2}{R_7} = 0 \Rightarrow V_2 - V_b = \frac{(V_1 - V_2) R_2}{R_7}
\]

\(\Rightarrow\) \( V_b = V_2 - \frac{(V_1 - V_2) R_2}{R_7} \Rightarrow (1 + \frac{R_2}{R_7}) V_2 - \frac{R_2}{R_7} V_1 \)

\(\Rightarrow\) KCL @ \( N_3 \) node:

\[
\frac{-V_3 - V_c}{R_4} + \frac{N_3}{R_6} = 0 \Rightarrow -\left( (R_3 - R_4) R_4 + (N_3)(R_4) \right) = 0
\]

\(\Rightarrow\) \( -N_3 R_4 + N_3 R_4 + N_3 R_4 = 0 \Rightarrow N_3(R_4 + R_4) = N_3(R_4 + R_4) \Rightarrow V_c = \frac{R_4 R_1}{R_4 + R_4} = V_d \)

\(\Rightarrow\) KCL @ \( N_4 \) node:

\[
\frac{-(V_4 - V_d)}{R_5} + \frac{(V_4 - V_d)}{R_5} = 0 \Rightarrow -(V_a - V_d) R_5 + (V_d - V_d) R_5 = 0
\]

\(\Rightarrow\) \( -V_a R_5 + V_d R_5 + V_d R_5 = 0 \Rightarrow V_d = \frac{R_5}{R_3} V_a + \left( 1 + \frac{R_5}{R_3} \right) V_d \)

\(\Rightarrow\) \( i_0 = \frac{V_d - V_d}{R_5} = \frac{N_C - N_C}{R_5} \Rightarrow \textcolor{red}{\text{plug in}} \)

\(\Rightarrow\) \( N_0 = \frac{-R_5}{R_3} \left[ (1 + \frac{R_5}{R_7}) V_r - \frac{R_5}{R_7} V_2 \right] + \left[ (1 + \frac{R_5}{R_3}) \right] \left[ \frac{R_6}{R_6 + R_4} \right] \left[ (1 + \frac{R_5}{R_7}) V_2 - \frac{R_5}{R_7} V_1 \right] \)

\(\Rightarrow\) \( i_0 = \frac{R_6}{R_5 + R_4} \left[ (1 + \frac{R_5}{R_7}) V_r - \frac{R_5}{R_7} V_1 \right] \left[ \frac{R_6}{R_3} \right] - \left[ \frac{R_5}{R_3} \left[ (1 + \frac{R_5}{R_7}) V_r - \frac{R_5}{R_7} V_2 \right] \right] \)

\(\Rightarrow\) \( i_0 = \frac{R_6}{R_5 + R_4} \left[ (1 + \frac{R_5}{R_7}) V_r - \frac{R_5}{R_7} V_1 \right] \left[ \frac{R_6}{R_3} \right] - \left[ \frac