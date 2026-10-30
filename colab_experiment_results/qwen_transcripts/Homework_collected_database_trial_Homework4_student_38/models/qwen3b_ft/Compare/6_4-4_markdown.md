![Source Image](6_4-4_source.png)
 KCL at top node 'art':
\[
\frac{V_1 - V_a}{R_1} + \frac{V_1 - V_2}{R_2} = 0
\]
\( V_a \) - voltage at top output

KCL at bottom node:
\[
\frac{V_2 - V_b}{R_3} + \frac{V_2 - V_1}{R_2} = 0
\]
\( V_b \) - voltage at bottom output

\[
V_o = V_a - V_b
\]

Rearrange for \( V_a \) and \( V_b \):
\[
V_a = \frac{R_1(V_1 - V_2)}{R_2} + V_1
\]
\[
V_b = \frac{R_3(V_2 - V_1)}{R_2} + V_2
\]

So,
\[
V_o = \frac{(R_1 + R_3)(V_1 - V_2)}{R_2} + V_1 - V_2
\]

