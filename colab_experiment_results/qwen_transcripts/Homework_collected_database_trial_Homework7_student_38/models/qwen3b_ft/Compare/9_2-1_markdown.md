![Source Image](9_2-1_source.png)
 KCL @ top node 'in': \( i_1 - \frac{v}{100} - C \frac{dv}{dt} = 0 \)

\( v = v_S - 2(i_1') - L \frac{di}{dt} \)
\[
\frac{dv}{dt} = -2 \frac{d(i_1)}{dt} - L \frac{d^2(i_1)}{dt^2}
\]

\( i_1 - \frac{1}{100}(v_S - 2i_1' - \left( \frac{di}{dt} \right)) - (L(-2 \frac{d(i_1)}{dt} - \left( \frac{d^2(i_1)}{dt^2} \right)) = 0 \)
\[
\textcolor{red}{10^{-6} \frac{d^2(i_1)}{dt^2} + 3 \cdot 10^{-5} \frac{d(i_1)}{dt} + 1.02i_1 = \frac{1}{100} v_S}
\]

