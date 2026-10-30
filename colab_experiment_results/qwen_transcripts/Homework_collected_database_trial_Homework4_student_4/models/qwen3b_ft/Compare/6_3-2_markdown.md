![Source Image](6_3-2_source.png)
 KVL for loop 1: \( 12V - (3k\Omega)i_1 - (2k\Omega)i_1 = 0 \)
\[
i_1 = \frac{12}{5} mA
\]

KCL at top node: 
\[
i_1 = i_0 = 2.4 mA
\]

KCL at mid. node: 
\[
-i_1 - i_2 = 0 \Rightarrow i_2 = -i_1 = -2.4 mA
\]

Ohm's at 1kΩ resistor: 
\[
(V_a - 0) = i_2(1k\Omega) \Rightarrow V_a = -2.4 V
\]

Ohm's Law at 4kΩ resistor: 
\[
(V_a - V_o) = i_0(4k\Omega)
\]
\[
\Rightarrow -2.4V - V_o = (2.4 mA)(4k\Omega)
\]
\[
\Rightarrow V_o = -12 V
\]

