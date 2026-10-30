![Source Image](8_3-5_source.png)
Before \( t = 0 \): (Switch closed, shorts the resistor)

\[
v_o(0) = 5V \quad (\text{Capacitor = open circuit})
\]

After \( t = 0 \):

KCL:

\[
-5V - iR = 5V - v_o
\]

\[
\Rightarrow -5V - 20k\Omega i = 5V - v_o
\]

\[
\Rightarrow v_o = 10V
\]

\[
R_T = 20k\Omega \quad (\text{resistor op amp & turn off voltage source})
\]

\[
\tau = 20k\Omega (4\, \mu F) = 80\, \mu s = 0.08\, s
\]

\[
v_o(t) = 10 - 5e^{-t/0.08}\, V
\]

