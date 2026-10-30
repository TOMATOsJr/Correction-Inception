![Source Image](9_4-2_source.png)
 

t ≥ 0: KVL @ top node 'in': \(\frac{1}{3}\frac{dV}{dt} - \frac{4}{3}V - i_L = 0\)  
i_L = \(\frac{1}{3}\frac{dV}{dt} - \frac{4}{3}V\)  

KVL right mesh: \(\frac{3}{4}(i_L - \frac{1}{3}\frac{dV}{dt}) + 1\frac{dV}{dt} = 0\)  

Operator method: \(i_L = \frac{1}{3}S^2V - \frac{4}{3}V\), \(\frac{3}{4}i_L - \frac{1}{4}S^2V + S i_L = 0\)  
\(\frac{1}{4}S^2V - V - \frac{1}{4}S^2V + \frac{1}{3}S^2V - \frac{4}{3}SV = 0\)  
\(\frac{1}{3}S^2V - \frac{4}{3}SV - V = 0\)  
\(S^2V - 4SV - V = 0\)  
\(S^2 - 4S - 1 = 0\)  
\(\alpha = -2\omega_0^2 - 1\)  
\(S_1 = -2 + \sqrt{3}\)  
\(S_2 = -2 - \sqrt{3}\)  

No source, no initial condition: complete response, \(V(t) = A_1 e^{(-2 + \sqrt{3})t} + A_2 e^{(-2 - \sqrt{3})t}\)  

With open, so \(\frac{dV(t)}{dt} = 0\):  
\(\frac{dV}{dt} = A_1(-2 + \sqrt{3})e^{(-2 + \sqrt{3})t} + A_2(-2 - \sqrt{3})e^{(-2 - \sqrt{3})t}\)  
\(A_1 + A_2 = 2\)  
\(\therefore (-2 + \sqrt{3})A_1 + (-2 - \sqrt{3})A_2 = 0\)  
\(\therefore V(t) = 2.155e^{-2.68t} - 1.155e^{-3.72t}, t \geq 0\)  

