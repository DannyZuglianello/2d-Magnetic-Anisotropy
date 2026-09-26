# 2D Anisotropic Magnetostatic Transformer Simulation

Este repositório contém os códigos para simulação 2D de magnetostática linear anisotrópica em núcleos de transformadores. O objetivo é analisar o impacto do aço de grão orientado, topologias de junta e erros de fabricação (ex: erro no ângulo de corte) na permeância.

## Conteúdo

* `generate_mesh.py`: Script Python/Gmsh para geração parametrizada da malha (controla dimensões, ângulo da junta).
* `ngsolve_solver.py`: Solver em Python utilizando a biblioteca NGSolve.
* `freefem_solver.edp`: Solver alternativo utilizando FreeFEM++.

## Dependências

* Python 3.x
* [Gmsh API](https://gmsh.info/) (`pip install gmsh`)
* [NGSolve](https://ngsolve.org/) (`pip install ngsolve`)
* [FreeFEM](https://freefem.org/) (binário `FreeFem++`)
* [ParaView](https://www.paraview.org/) (para visualização dos resultados)

## Como Executar

**1. Gerar a malha (.msh):**
```bash
python generate_mesh.py
```
*(O Solver em Python já realiza a geração de malhas. Use para reproduzir os resultados em FreeFem).*

**2. Rodar o Solver:**

Via Python/NGSolve:
```bash
python ngsolve_solver.py
```

Via FreeFEM:
```bash
FreeFem++ freefem_solver.edp
```

**3. Visualização:**
Ambos os solvers exportam arquivos `.vtu`. Basta abri-los no **ParaView** para visualizar as linhas e intensidade da densidade de fluxo magnético.
