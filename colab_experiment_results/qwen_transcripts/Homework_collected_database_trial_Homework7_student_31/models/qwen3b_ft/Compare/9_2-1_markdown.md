![Source Image](9_2-1_source.png)
 

KCL:

\[
i_L' = \frac{v}{R_1} + C \frac{dv}{dt}
\]

\(\Rightarrow\) current going to inductor

\textcolor{red}{KVL:}

\[
v_R = R_1 i_L' + L \frac{di_L}{dt} + v
\]

\[
\frac{di_L}{dt} = \frac{1}{R_2} \frac{dv}{dt} + (C \frac{d^2 v}{dt^2})
\]

Plug everything into KVL:

\[
v_R = R_1 \left[ \frac{v}{R_2} + (C \frac{dv}{dt}) \right] + L \left[ \frac{1}{R_2} \frac{dv}{dt} + (C \frac{d^2 v}{dt^2}) \right] + v
\]

\[
v_R = (LC) \frac{d^2 v}{dt^2} + (R_1 C + \frac{L}{R_2}) \frac{dv}{dt} + \left( \frac{R_1}{R_2} + 1 \right) v
\]

\[
v_R = (1 \times 10^{-8}) \frac{d^2 v}{dt^2} + (3 \times 10^5) \frac{dv}{dt} + 1.02 v
\]

\[
\Rightarrow \left[ \frac{d^2 v}{dt^2} + 3000 \frac{dv}{dt} + (1.02 \times 10^9) v \right] = (1 \times 10^8) v_R
\]

