![Source Image](6_4-5_source.png)
 

\[
v_3 = v_c \quad i_3 = \frac{v_3 - v_0}{R_3}
\]

\[
KCL @ node d 'in': \quad \frac{v_a - v_3}{R_3} + \frac{v_3 - v_2}{R_5} = 0 \quad \Rightarrow \quad v_0 = \frac{R_5(v_a - v_2)}{R_3} + v_3
\]

\[
KCL below R_1 'out': \quad \frac{v_1 - v_a}{R_1} + \frac{v_1 - v_2}{R_7} = 0 \quad \Rightarrow \quad v_a = \frac{R_1(v_1 - v_2)}{R_7} + v_1
\]

\[
KCL @ node a 'out': \quad \frac{v_a - v_1}{R_1} + \frac{v_a - v_d}{R_3} = 0 \quad \Rightarrow \quad v_1 - v_2 + \frac{R_1(v_1 - v_2) + R_3 v_1 - R_7 v_2}{R_3 R_7} = 0
\]

\[
v_3 = \frac{(R_1 + R_3)(v_1 - v_2) + R_7 v_1}{R_7}
\]

\[
v_e - v_3 = \frac{R_1(v_1 - v_2) + R_3 v_1 - (R_1 + R_3)(v_1 - v_2) - R_7 v_1}{R_7} = \frac{R_3(v_2 - v_1)}{R_7}
\]

\[
v_0 = \frac{R_5(v_a - v_2)}{R_7} + v_3 = \frac{R_5(v_2 - v_1)}{R_7} + \frac{(R_1 + R_3)(v_1 - v_2) + R_7 v_1}{R_7} = \frac{R_3(v_2 - v_1)}{R_7}
\]

\[
v_3 - v_0 = \frac{R_5(v_2 - v_1)}{R_7}
\]

