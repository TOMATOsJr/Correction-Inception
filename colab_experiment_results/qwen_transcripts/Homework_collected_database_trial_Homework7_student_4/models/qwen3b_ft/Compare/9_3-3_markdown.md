![Source Image](9_3-3_source.png)
 

(KVL: \( v_c = (2 \textcolor{red}{\pi}) i_L + (10^{-3} \textcolor{red}{H}) \frac{di_L}{dt} \))

(KCL: \( \frac{v_s - v_c}{1 \textcolor{red}{\Omega}} = i_L + (10^{-4} \textcolor{red}{F}) \frac{d i_L}{dt} \))

\[
v_s - 2 \textcolor{red}{i_L} - (10^{-3} \textcolor{red}{H}) \frac{di_L}{dt} = i_L + (10^{-4} \textcolor{red}{F}) \left( (2 \textcolor{red}{\pi}) \frac{di_L}{dt} + (10^{-3} \textcolor{red}{H}) \frac{d^2 i_L}{dt^2} \right)
\]

\[
\therefore v_s = (10^{-3} \textcolor{red}{H}) \frac{d^2 i_L}{dt^2} + (10^{-3} \textcolor{red}{F}) \frac{d i_L}{dt} + 3 \textcolor{red}{i_L}
\]

\[
v_s = (10^{-3} \textcolor{red}{H}) s^2 + (10^{-3} \textcolor{red}{F}) s + 3 \textcolor{red}{i_L}
\]

\[
(10^{-3} \textcolor{red}{H}) v_s = (s^2 + (10^{-3} \textcolor{red}{F})) s + (3 \cdot 10^8) \textcolor{red}{i_L}
\]

\[
\therefore \textcolor{red}{i_L} = s^2 + (10^{-3} \textcolor{red}{F}) s + (3 \cdot 10^8) = 0
\]

\[
s_{1,2} = \frac{-(10^{-3} \textcolor{red}{F}) \pm \sqrt{(10^{-3} \textcolor{red}{F})^2 - 4(3)(3 \cdot 10^8)}}{2} = \textcolor{red}{-3.031 \cdot 10^3}, \quad s_{2} = \textcolor{red}{-9.897 \cdot 10^4}
\]

\[
\therefore s_1 = \textcolor{red}{-3031.260}, \quad s_2 = \textcolor{red}{-98968.740}
\]

