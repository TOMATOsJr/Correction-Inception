![Source Image](6_4-6_source.png)
Due to property of ideal op-amp: \( V_O = 0 \textcolor{red}{V} \)

LL @ a: 
\[
-12V - V_A = \frac{V_A - V_C}{40k\Omega} + \frac{V_A}{20k\Omega} + \frac{V_C - V_R}{10k\Omega}
\]
\[
-12 - V_A = 2V_A - 2V_C + V_A + 4V_A - 4V_C
\]
\[
-12 - V_A = 7V_A - 4V_C
\]
\[
8V_A = 4V_C - 12 \quad \Rightarrow \quad V_A = \frac{1}{2}V_C - \frac{3}{2}
\]

KCL @ b: 
\[
V_A - V_B = 0 = \frac{V_A^2 - V_C}{20k\Omega} \quad \Rightarrow \quad 5V_A = -4V_C
\]
\[
V_A = -\frac{4}{5}V_C
\]
\[
\Rightarrow -\frac{4}{5}V_C = \frac{1}{2}V_C - \frac{3}{2} \quad \Rightarrow \quad V_C = -\frac{4}{5}\left(\frac{15}{13}\right)
\]
\[
-8V_C = 5V_C - 15 \quad \Rightarrow \quad -13V_C = -15 \quad \Rightarrow \quad V_C = \frac{15}{13}V
\]

\[
V_A = -\frac{12}{13}V
\]

