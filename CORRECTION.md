# Guide simple de correction

Ce guide donne l'ordre exact à suivre pour préparer et tester le projet.
Toutes les commandes sont à lancer depuis la racine du clone `RAG-42`.

## 1. Arborescence attendue

Avant la correction, les fichiers doivent être rangés ainsi :

```text
RAG-42/
├── src/
├── test/
├── datasets_public/
├── exams/
│   └── scripts/
│       ├── exam_retrieval.sh
│       ├── exam_edge_cases.sh
│       └── exam_answer.sh
├── data/
│   ├── raw/
│   │   └── vllm-0.10.1/
│   │       ├── docs/
│   │       ├── vllm/
│   │       └── ...
│   └── datasets/
│       └── private/
│           ├── AnsweredQuestions/
│           │   ├── dataset_docs_private.json
│           │   └── dataset_code_private.json
│           └── UnansweredQuestions/
│               ├── dataset_docs_private.json
│               └── dataset_code_private.json
├── moulinette-ubuntu
├── moulinette-docker.sh
├── Makefile
├── pyproject.toml
└── uv.lock
```

Ne déplacez pas `datasets_public/` : il est déjà correctement placé dans le
dépôt. `data/`, `exams/`, les datasets privés et le binaire moulinette sont
ignorés par Git et ne seront pas envoyés avec le projet.

## 2. Cloner le projet

Depuis GitHub :

```bash
git clone git@github.com:Mind-alia/RAG-42.git
cd RAG-42
```

Pour la correction officielle, le clone Vogsphere peut être utilisé à la
place du clone GitHub.

## 3. Importer les fichiers de correction

### Méthode recommandée : utiliser le kit complet

Placez `rag-correction-mac-docker.tar.gz` dans le dossier parent de `RAG-42`,
puis lancez depuis la racine du projet :

```bash
tar -xzf ../rag-correction-mac-docker.tar.gz
mkdir -p data/raw data/datasets
ln -s ../../vllm-0.10.1 data/raw/vllm-0.10.1
ln -s ../../private data/datasets/private
```

Le kit contient :

- `vllm-0.10.1/` ;
- les datasets privés dans `private/` ;
- `moulinette-ubuntu` ;
- les scripts dans `exams/` ;
- `moulinette-docker.sh` pour macOS.

Si un lien existe déjà, ne relancez pas la commande `ln`. Vérifiez simplement
les chemins avec les commandes de la section suivante.

### Méthode manuelle : fichiers importés séparément

Rangez chaque élément à l'emplacement suivant :

| Élément importé | Emplacement final |
|---|---|
| dépôt `vllm-0.10.1` | `data/raw/vllm-0.10.1/` |
| dossier privé | `data/datasets/private/` |
| scripts d'examen | `exams/scripts/` |
| binaire Ubuntu | `moulinette-ubuntu` à la racine |
| lanceur Docker | `moulinette-docker.sh` à la racine |

Si vLLM est fourni sous forme d'archive :

```bash
mkdir -p data/raw
unzip /chemin/vers/vllm-0.10.1.zip -d data/raw
```

Si les datasets privés sont fournis sous forme d'archive :

```bash
mkdir -p data/datasets
unzip /chemin/vers/datasets_private.zip -d data/datasets
```

Après extraction, le chemin final doit être exactement
`data/datasets/private/AnsweredQuestions`, et non un dossier privé imbriqué
deux fois.

## 4. Vérifier le rangement

```bash
test -d data/raw/vllm-0.10.1/docs && echo "vLLM docs: OK"
test -d data/raw/vllm-0.10.1/vllm && echo "vLLM code: OK"

test -f data/datasets/private/AnsweredQuestions/dataset_docs_private.json \
  && echo "Dataset docs répondu: OK"
test -f data/datasets/private/AnsweredQuestions/dataset_code_private.json \
  && echo "Dataset code répondu: OK"
test -f data/datasets/private/UnansweredQuestions/dataset_docs_private.json \
  && echo "Dataset docs questions: OK"
test -f data/datasets/private/UnansweredQuestions/dataset_code_private.json \
  && echo "Dataset code questions: OK"

test -f moulinette-ubuntu && echo "Moulinette: OK"
test -f exams/scripts/exam_retrieval.sh && echo "Scripts examen: OK"
```

Les huit lignes doivent afficher `OK`.

Rendez les outils exécutables :

```bash
chmod +x moulinette-ubuntu
chmod +x moulinette-docker.sh
chmod +x exams/scripts/*.sh
```

Vérifiez enfin que les fichiers locaux de correction ne sont pas proposés au
commit :

```bash
git status
```

Le dépôt doit rester propre.

## 5. Installer et vérifier le projet

Exécutez dans cet ordre :

```bash
make install
make test
make lint
```

Résultats attendus :

- `77 passed` pour les tests ;
- aucune erreur flake8 ;
- aucune erreur mypy.

Le contrôle strict facultatif peut aussi être lancé :

```bash
make lint-strict
```

## 6. Créer l'index

```bash
uv run python -m src index --max_chunk_size 2000
```

Résultat attendu :

```text
Indexed 974 documents into 9347 chunks.
```

## 7. Tester une recherche simple

```bash
uv run python -m src search \
  "How to configure OpenAI server?" \
  --k 10
```

La commande doit afficher une liste de sources avec leurs chemins et leurs
indices de caractères.

## 8. Lancer la correction du retrieval

### Sur un Mac Apple Silicon

Démarrez Docker Desktop :

```bash
open -a Docker
docker info
```

Lancez ensuite :

```bash
bash exams/scripts/exam_retrieval.sh \
  --student-path . \
  --moulinette-path ./moulinette-docker.sh
```

Résultats attendus :

```text
FINAL RESULT: 4/4
STATUS: PASS
Docs Recall@5 = 0.850
Code Recall@5 = 0.640
```

### Sur Linux x86-64

Docker n'est pas nécessaire. Utilisez directement le binaire :

```bash
bash exams/scripts/exam_retrieval.sh \
  --student-path . \
  --moulinette-path ./moulinette-ubuntu
```

## 9. Tester les cas limites

Cette commande est identique sur macOS et Linux :

```bash
bash exams/scripts/exam_edge_cases.sh --student-path .
```

Résultat attendu :

```text
FINAL RESULT: 4/4
STATUS: PASS
```

## 10. Tester la qualité des réponses

Sur macOS :

```bash
bash exams/scripts/exam_answer.sh \
  --student-path . \
  --moulinette-path ./moulinette-docker.sh
```

Sur Linux, remplacez `./moulinette-docker.sh` par
`./moulinette-ubuntu`.

Lorsque le script demande trois questions, saisissez une question à chaque
invite :

```text
What vLLM metric tracks the number of requests currently being processed in model execution batches?
```

```text
What is the CLI command to start vLLM serve?
```

```text
What does the LLM.chat method in vLLM implement?
```

Les trois réponses ont été vérifiées comme satisfaisantes.

## 11. Ordre résumé à retenir

```text
1. Cloner le projet
2. Importer vLLM, les datasets privés, la moulinette et exams/
3. Vérifier le rangement
4. make install
5. make test
6. make lint
7. Créer l'index
8. Lancer exam_retrieval.sh
9. Lancer exam_edge_cases.sh
10. Lancer exam_answer.sh
```

## Seuils obligatoires

- Indexation : 5 minutes maximum.
- Retrieval de 1000 questions : 90 secondes maximum.
- Recall@5 docs : au moins 80 %.
- Recall@5 code : au moins 50 %.
- Qualité des réponses : au moins 2 réponses satisfaisantes sur 3.
