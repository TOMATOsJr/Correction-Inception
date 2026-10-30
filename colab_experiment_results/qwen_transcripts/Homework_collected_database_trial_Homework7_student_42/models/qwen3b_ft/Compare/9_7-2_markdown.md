![Source Image](9_7-2_source.png)
 Defining Normal Response

\[
- v_S C(t) + 7 C \left( \frac{d v_C}{dt} \right) + 0.1 C \left( \frac{d^2 v_C}{dt^2} + v_C(t) \right) = 0
\]

\[
\textcolor{red}{(8.333 \times 10^9)} \frac{d^2 v_C}{dt^2} + 0.005833 \frac{dv_C}{dt} + v_C(t) = v_S C(t)
\]

\[
v_C(t) = e^{-35t} \left( C_1 \cos(103.805 t) + C_2 \sin(103.805 t) \right)
\]

a) \( v_S = 2 V \)

\[
v_C(t) = A \sin \textcolor{red}{(2t)} \quad \text{when plug in, } A = 2.
\]

\[
\Rightarrow v_C(t) = 2 V
\]

b) \( v_S = 0.2 V \)

\[
\textcolor{red}{4.11A = 4t + B} \quad 0.005833 |A| + tB = 0.2t
\]

\[
v_C(t) = A
\]

\[
\Rightarrow A = 0.2
\]

\[
v_C(t) = A
\]

\[
\textcolor{red}{4t + B = 0} \quad \Rightarrow B = -0.00113
\]

\[
\Rightarrow v_C(t) = \sqrt{0.24 - 0.00113} V
\]

c) \( v_S = e^{300t} V \)

\[
v_C(t) = Ae^{-20t}
\]

\[
v_C(t) = 400A e^{-20t} + 0.074997 A e^{-30t} - 0.174993 A e^{-20t} + 0.136t e^{-20t}
\]

\[
\Rightarrow v_C(t) = \left[ 10 - 30A e^{-20t} \right] V
\]

\[
\textcolor{red}{v_C(t) = 0.9A e^{-20t}} \quad A = 10^{-4}
\]

