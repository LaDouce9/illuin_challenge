# Dictionnaire des Variables - Challenge Illuin
## Projet : Classification Multi-label d'Exercices d'Algorithmique

Ce document répertorie et explique l'utilité de chaque variable présente dans le dataset **xCodeEval** utilisé pour ce challenge.

---

## 1. La Variable Cible (Target)
C'est l'objectif final du modèle.

| Variable | Type | Description |
| :--- | :--- | :--- |
| **`tags`** | `List[str]` | **Variable Cible**. Liste des concepts algorithmiques nécessaires pour résoudre le problème. Le challenge se concentre sur 8 tags prioritaires : `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`, `games`, `probabilities`. |

---

## 2. Variables de l'Énoncé (Input Features)
Ces variables sont disponibles dès la lecture du problème. Elles constituent la source principale d'information pour comprendre "ce qu'il faut faire".

| Variable | Description | Utilité pour le Feature Engineering |
| :--- | :--- | :--- |
| `prob_desc_description` | Texte complet de l'énoncé. | **Essentielle**. Analyse NLP pour extraire des mots-clés algorithmiques et le contexte. |
| `prob_desc_input_spec` | Format des données d'entrée. | Indices sur les structures de données (ex: "n nodes", "m edges" → `graphs`). |
| `prob_desc_output_spec` | Format de la réponse attendue. | Indices sur le type de calcul (ex: "modulo 10^9+7" → `math` ou `number theory`). |
| `prob_desc_sample_inputs` | Exemples d'entrées. | Visualisation de la structure brute des données. |
| `prob_desc_sample_outputs` | Exemples de sorties. | Compréhension du résultat attendu. |
| `prob_desc_notes` | Explications sur les exemples. | Détails subtils sur la logique (Attention : 27% de valeurs manquantes). |
| `difficulty` | Score de difficulté (800-3500). | **Forte corrélation**. Certains tags sont intrinsèquement plus difficiles que d'autres. |
| `prob_desc_time_limit` | Limite de temps (ex: 1.0s). | Indique si une solution optimisée (ex: O(n log n)) est requise. |
| `prob_desc_memory_limit` | Limite de mémoire (ex: 256MB). | Indique des contraintes sur les structures de données utilisées. |

---

## 3. Variables de la Solution (Solution Features)
Ces variables concernent la réponse fournie (le code). Elles permettent de comprendre "comment le problème a été résolu".

| Variable | Description | Utilité pour le Feature Engineering |
| :--- | :--- | :--- |
| **`source_code`** | Code Python de la solution. | **Majeure**. Analyse de la structure (boucles, récursion), des imports (`math`, `collections`) et des patterns. |
| `lang` | Version de Python. | Feature catégorielle (Python 3, PyPy 3, etc.). |
| `exec_outcome` | Résultat d'exécution. | **Leakage potentiel**. Généralement `PASSED`. À utiliser pour filtrer le train set, mais pas comme feature d'entrée. |

---

## 4. Métadonnées Techniques
Variables servant à l'organisation du dataset, sans valeur prédictive.

| Variable | Description |
| :--- | :--- |
| `src_uid` | ID unique de l'énoncé (problème). |
| `code_uid` | ID unique de la solution (code). |
| `prob_desc_created_at` | Timestamp de création du problème. |
| `file_name` | Nom du fichier source d'origine. |

---

## 5. Synthèse Stratégique

### Inférence (Usage Réel)
Pour prédire les tags d'un nouveau problème, le modèle s'appuiera sur :
1.  **Le texte** (`description`, `input_spec`, `output_spec`)
2.  **Les contraintes** (`difficulty`, `time_limit`)
3.  **Le code** (`source_code`) si une solution est déjà disponible.

### Apprentissage (Training)
Le modèle apprend la fonction :
> $f(\text{Énoncé} + \text{Code}) = \text{Tags}$

### Points d'Attention
- **LaTeX** : Les descriptions contiennent des formules mathématiques (ex: `$$$n \le 10^5$$$`) qui sont des indicateurs forts pour les tags `math`.
- **Imports** : La présence de `import heapq` est un signal quasi-certain pour `graphs` ou `trees`.
- **Déséquilibre** : Certains tags sont beaucoup plus fréquents que d'autres, ce qui nécessite une attention particulière lors de l'entraînement.
