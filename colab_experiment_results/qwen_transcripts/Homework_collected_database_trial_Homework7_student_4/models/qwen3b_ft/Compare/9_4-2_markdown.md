![Source Image](9_4-2_source.png)
 

\[
v(0) = 2V \quad i(0) = 0A
\]

\[
\textcolor{red}{\frac{C}{R}} \quad \textcolor{red}{\frac{t}{R}} \quad \textcolor{red}{\frac{R}{2\pi R}} \quad \textcolor{red}{\frac{t}{R}}
\]

\[
\Rightarrow v(t) = L \cdot \frac{d\textcolor{red}{v(t)}}{dt} \quad \Rightarrow i = \frac{1}{R} v(t) \quad \Rightarrow i = -L \cdot s \cdot v(t) - \frac{1}{R} v(t)
\]

\[
v(t) = L \cdot \frac{d\textcolor{red}{v(t)}}{dt} \quad \Rightarrow v(t) = L \cdot s \left(-L \cdot s \cdot v(t) - \frac{1}{R} v(t)\right)
\]

\[
v(t) = -L \cdot s \cdot s^2 v(t) - \frac{L}{R} \cdot s v(t)
\]

\[
\therefore v(t) \left(L \cdot s^2 + \frac{L}{R} s + 1\right) \Rightarrow 0 = s^2 + \frac{1}{LC} s + \frac{1}{LC} \quad \text{or forced response = 0}
\]

\[
\therefore s^2 + 4s + 3 = 0
\]

\[
\therefore (s + 3)(s + 1) \quad ; s_1 = -3, s_2 = -1
\]

\[
v(t) = A_1 e^{-3t} + A_2 e^{-t} \quad \Rightarrow v(0) = 2 = A_1 + A_2
\]

\[
-3A_1 - A_2 = -\frac{i(0)}{RC} \quad \Rightarrow -8 = -3A_1 - A_2
\]

\[
-6 = -2A_1 \quad \Rightarrow A_1 = 3
\]

\[
2 = 3 + A_2 \quad \Rightarrow A_2 = -1
\]

\[
\therefore v(t) = 3e^{-3t} - e^{-t}V
\]

