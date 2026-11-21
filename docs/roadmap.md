# RustBlock — Roadmap fasi 2–8

**Stato di riferimento:** 8 novembre 2026 → aggiornato **21 novembre 2026**
**Fase 1:** completata  
**Fase 8:** implementazione completata nell'ambito dichiarato; confronto numerico completo AWQ ancora aperto  
**Fasi 2–7 e 5.2–5.4:** implementazione parziale; le parti presenti nel codice sono indicate come parziali finché manca l'evidenza di accettazione esterna.

**Aggiornamento del 21 novembre 2026 — revisione a fondo.** Ho revisei tutto il progetto (core, CLI, GUI, docs) e tenuto aggiornata l'evidenza di oggi:

- **Miglioramenti applicati**
  - `ops`: `metadata_set` e `prune` rifiutano l'invocazione con input == output (impediscono che il writer sostituisca il file di origine); il test conferma che l'input resta intatto.
  - `download`: probe HEAD con fallback a `Range: bytes=0-0` per i server che non rispondono al HEAD (o rispondono non-2xx).
  - `quality`: escaping CSV corretto per i nomi tensore (virgole, virgolette, nuove righe non spezzano più le righe del CSV).
  - CLI `inspect`: conteggio corretto dei target nativi (21).
  - GUI: i job DONE mostrano ora il nome del file di output; help modale con voci nuove (cache Discover, badge download, “Non so cosa scegliere”); schermata vuota con i suggerimenti sulle scorciatoie.
- **Verifiche di oggi**: `cargo test --workspace` passa (crates core 42, gui 17 — `every_surface_renders_without_panicking` ora copre 3 tab + Inspector e il nuovo `log_level_filter_matches_only_selected_levels` passa —, transcode 16, end_to_end 6, advanced 8; parità ignorata come di sé); `cargo fmt --all -- --check` e `cargo clippy --workspace --all-targets -- -D warnings` passano.
- **Sei test aggiunti**: `ops` (3: rifiuto same path per `metadata_set` e `prune`, preservazione input per `metadata_set`), `download` (fallback probe range-only), `quality` (2: escaping CSV).

**Aggiornamento del 25 novembre 2026 — GUI riscritta da zero.** Con l'evidenza del 21 nov ho riletto la GUI e riprodotto architettura e viste da zero (`tab: Tab` → Convert/Discover/**Inspector**, `selected_job`, `logs: Vec<LogEntry>` con filtri per livello, `benchmark: Option<BenchmarkUi>`):

- **Convert (nuova architettura):** header con riepilogo live (`N files · X ready · Y running · Z done · W failed`), banner benchmark, barra di batch con barra di progresso e botone Stop, card job con nome selezionabile, badge fit, progresso inline (speed/ETA), card Done con nome dell'output e delta X → Y, errori in rosso; pannello dettagli sulla destra (fit breakdown, fatti, percorso output, azioni).
- **Inspector (nuova scheda):** fatti (versione, allineamento, dati, tipo file), metadati (fino a 12 chiavi), istogramma dei tipi tensore con barre (tipo, count, byte, barra di proporzione) e stima di memoria (weights/kv/overhead/total).
- **Benchmark (nuovo):** pulsante “Benchmark” nell'header Convert → `llama-bench -m <model> -ngl 99` sul modello finito più grande; il risultato va nel log. Chiude il gap rispetto a `quick-test` sola.
- **Discover:** card con licenza (`License: MIT`) — campo `license` letto dall'API Hub; card più informativi (`gguf_files`, `downloads`, `last_modified`).
- **Log:** filtri per livello (I/✓/⚠/✗), conteggio `visible/total`, filtro testo, Clear, Close.
- **Test:** `every_surface_renders_without_panicking` ora copre i 3 tab + Inspector; test nuovi: `log_level_filter_matches_only_selected_levels` (passa).

I gate esterni che restano aperti sono elencati nella sezione “Lavoro aperto”; nessun gate è bloccato dal codice.

Questo documento raccoglie il piano approvato, lo stato reale dell'implementazione e i criteri che devono essere soddisfatti prima di dichiarare una fase conclusa. Le stime in settimane sono quelle del piano originale e servono come ordine di grandezza, non come impegno di consegna.

## Legenda

| Simbolo | Significato |
|---|---|
| ✅ | Completato e verificato |
| ◐ | Parzialmente presente, ma i criteri di accettazione non sono ancora soddisfatti |
| ☐ | Pianificato, non implementato |
| ⛔ | Bloccato da una dipendenza esterna o da una decisione di prodotto |

## Stato sintetico

| Fase | Settimane | Stato | Sintesi |
|---|---:|---|---|
| 1 — Motore di ricette | 1–8 | ✅ | Preset misti, override, protezioni, CLI/GUI e test |
| 2 — imatrix | 9–11 | ◐ | Parser legacy/GGUF, override pesati, CLI e generatore esterno; fixture reali non disponibili |
| 3 — Qualità misurabile | 12–15 | ◐ | RMSE/SNR/coseno, ranking, report JSON/CSV, planner budget corretto nei fallback e KLD via `llama-perplexity`; test su 7B e 10 modelli non eseguito |
| 4 — Operazioni GGUF | 16–18 | ◐ | Split/merge con manifest, verifica SHA-256, comandi operativi, compilazione Jinja con minijinja, validazione rope/chat-template e rendering del template con un contesto JSON presenti; il controllo delle coppie rope contro il runtime effettivo resta esterno |
| 5 — Stima fit RAM/VRAM | 19–20 | ◐ | Pesi, KV GQA per-layer, sliding window, MLA e stato SSM, hardware, badge fit verde/giallo/rosso e suggerimento a budget; confronto reale con llmvram/hf-mem non eseguito |
| 6 — Discover e installazione | 21–24 | ◐ | CLI download 16 range/file e 8 file, eventi JSON-lines, mmproj, cache con quota, Ollama/LM Studio hand-off, quick test; GUI usa download parallelo, projector esplicito, hash LFS e riuso cache locale; resume reale verificato con hash della Hub |
| 7 — Famiglia Prism | 25–27 | ◐ | Reblock, auto-detect Q2_0, PTQ1 rifiutato e metadati `prism.*` preservati byte per byte; reblock reale 2B caricato, ma verifica numerica MTP/Hadamard non eseguita |
| 8 — safetensors → GGUF | tardiva | ◐ | Bridge esterno; dequant nativo FP8/AWQ; conversione streaming per LLaMA/Mistral/Mixtral/Qwen2/Qwen3 dense e MoE con BPE o SentencePiece, verificata con llama.cpp; confronto AWQ tensore-per-tensore aperto |
| 5.2 — Guide in-app | trasversale | ◐ | Help F1, popover “?” sui parametri, badge fit verde/giallo/rosso, suggerimento a budget, tooltip; GUI riscritta da zero il 25 nov 2026 (inspector, benchmark, log filters); test con 5 utenti non eseguito |
| 5.3 — Feedback e prestazioni | trasversale | ◐ | Log filtrabile, progress dettagliati, UI non bloccante, Ctrl+O/Ctrl+S/Ctrl+L/F1; avvio 0,22–0,43 s, interazione con 5 utenti non eseguita |
| 5.4 — Criteri UI | trasversale | ◐ | WCAG AA verificato da test automatico e avvio <1 s misurato; test con 5 utenti non eseguito |

---

## Lavoro aperto (gate esterni)

I criteri rimasti richiedono ambienti, modelli o dati che non possono produrre qui. Niente di questo è bloccato dal codice: ogni sezione sopra ha implementato l'80–100% della propria area di influenza.

| Area | Gate aperto | Tipo di gate |
|---|---|---|
| Fase 2 (imatrix) | Fixture reali provenienti da bartowski/unsloth (o simili) per verificare che MSE/SNR migliorino rispetto alla stessa quantizzazione senza imatrix | Modelli/dati reali (Phase 3) |
| Fase 2 | Target IQ* usabili con pmetal/llama.cpp | Dipendenza esterna (prevede un encoder IQ completo) |
| Fase 3 | Test su 7B e confronto su 10 modelli; soglia SNR esplicita esterna | Modelli reali (7B/10 modelli) |
| Fase 4 | Controllo delle coppie rope e dei tipi del chat template contro il runtime effettivo (per avvertenza compatibilità) | Runtime esterno |
| Fase 5 | Confronto reale con `llmvram` e `hf-mem` su un set di 10 modelli; caso 70B su GPU 24 GB | Modelli e tool esterni |
| Fase 6 | Installazione verificata in ambiente Ollama/LM Studio reali (applicazioni installate) | Dipendenza esterna |
| Fase 7 | Reblock di 5 file Prism PQ2_0 reali: verifica numerica MTP/Hadamard su file reali | Runtime esterno + prova di accettazione (fixture reali) |
| Fase 8 | Confronto AWQ tensore-per-tensore (fino a differenze numeriche note) | Modelli reali + tool esterni |
| 5.2 | Test di usabilità con 5 utenti non esperti che completano il flusso “scarica e installa un modello” | Sessioni utente reali |
| 5.4 | Test di usabilità con 5 utenti non esperti; la misura avvia di oggi è 0,22–0,43 s | Sessioni utente reali |

Per questi gate il codice deve mantenere diverse property, e ogni nuova modifica al codice deve preservarle:

- evidenza di accettazione documentata per ogni sezione (test automatici, output del CLI, scenari eseguiti);
- limiti non verificati indicati esplicitamente (se qualcosa non lo è, va scritto qui);
- rate limited per tool esterni (`llama-imatrix`, `llama-perplexity`, `llama-cli`, `llama-server`, `llama-quantize`) — rilevamento PATH, mai download automatico.

---

## Fase 1 — Motore di ricette (completata)

### Implementato

- Motore di ricette in `crates/core/src/recipe.rs` con preset `Q2_K`, `Q3_K_M`, `Q4_K_M`, `Q5_K_M`.
- Regole llama.cpp per `attn_v`, `ffn_down`, `attn_output`, output/tied embeddings, GQA e `use_more_bits`.
- Override regex e da file, sempre prioritari sul preset; primo match vince.
- Protezioni per norme, router MoE, tensori SSM/ibridi e tensori non quantizzabili.
- Fallback per shape incompatibili: `Q4_K → Q5_0`, `Q5_K → Q5_1`, `Q6_K → Q8_0`, `Q2_K/Q3_K/TQ → Q4_0`, poi `F16`.
- Target uniformi `Q3_K_S`, `Q4_K_S`, `Q5_K_S`; alias CLI `q3_k`, `q4_k`, `q5_k` verso i preset `_M`.
- Requantizzazione non silenziosa: default rifiutato, `--allow-requantize` esplicito e conferma nella GUI.
- CLI con `--tensor-type`, `--tensor-type-file`, `--dump-tensor-types`, `--allow-requantize`.
- GUI con griglia a cinque colonne, preset visibili e default `Q4_K_M` per i modelli float.
- Ricette riallineate a `llama-quant.cpp` b12359: `use_more_bits` con divisione intera `7*n/8` (prima l'ultimo ottavo era `n - n/8`), preset `_S` uniformi (solo output/tied embedding a `Q6_K`, `Q4_K_S` che promuove le prime quattro `attn_v` a `Q5_K`), regole `ffn_down` basate sul conteggio dei tensori `ffn_down` (corretto anche per i MoE) e `Q4_K_S` con la soglia `layer < n_ffn_down/8`.
- Harness differenziale `quantize_parity` e job CI `parity` contro la build pinnata b12359.
- Test golden, override, MoE/SSM, embedding legati, fallback shape, parser, requantizzazione ed end-to-end.
- README e screenshot aggiornati.

### Limiti noti

- La parità **byte-per-byte** dell'output non è verificata: l'harness confronta la mappa dei tipi per tensore, non header, offset e payload. Il gate dichiarato nella metrica di successo (§7) resta quindi aperto su quel punto.
- **Evidenza 2026-11-19:** l'harness (`crates/core/tests/quantize_parity.rs`) è stato eseguito contro la build pinnata llama.cpp **b12359** (commit b1b770d90) e passa per `Q2_K`, `Q3_K_S/M`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K` e `Q8_0`: la mappa dei tipi è identica a `llama-quantize` su un fixture F16 di 8 layer. Il job CI `parity` scarica la stessa release e ripete il confronto.
- Le fasi successive non sono incluse nella Fase 1.

---

## Fase 2 — imatrix (settimane 9–11)

**Stato:** ◐ parziale; parser, applicazione dei pesi, CLI e generazione esterna presenti; mancano fixture reali e target IQ.
**Dipendenze:** parser imatrix, quantizzatori con pesi, encoder IQ (per la parte `IQ*`), fixture reali, metrica di qualità della Fase 3.

### Obiettivi

- Leggere matrici di importanza in formato `.imatrix` legacy e `.gguf`.
- Applicare i pesi ai tipi che li supportano: `IQ*`, `Q2_K`, `Q3_K`.
- Generare una imatrix solo se `llama-imatrix` è disponibile, senza reimplementare l'inferenza.
- Mostrare un avviso sui rischi di overfitting delle calibrazioni Wikipedia.

### Lavoro previsto

1. Parser dei due formati con validazione di versione, dimensione e nomi dei tensori.
2. Associazione fra nomi imatrix e nomi dei tensori GGUF, con errore esplicito per tensori mancanti o ambigui.
3. Passaggio dei pesi ai quantizzatori supportati, mantenendo il comportamento attuale quando la matrice non è presente.
4. Comando/opzione CLI per caricare la matrice e riportare nel JSON quali tensori l'hanno ricevuta.
5. Rilevamento di `llama-imatrix` tramite `PATH` o variabile esplicita; se assente, funzione disabilitata con messaggio chiaro. Nessun download o bundling automatico.
6. Avviso UI e documentazione sul fatto che una calibrazione Wikipedia può non rappresentare il dominio d'uso.

### Criteri di accettazione

- Test con imatrix reali (per esempio fixture provenienti da bartowski/unsloth): MSE e SNR migliorano rispetto alla stessa quantizzazione senza imatrix. → **Gate esterno** (Phase 3), vedi “Lavoro aperto”.
- Se `llama-imatrix` non è trovato, la generazione è disabilitata e il messaggio spiega come installarlo o come fornire una matrice esistente.
- La lettura di una matrice malformata fallisce senza modificare il file di output.
- La mappa dei pesi applicati è riportata nel report e nel dump.

### Blocchi e note di scope

- RustBlock oggi **non scrive target IQ**: la parte `IQ*` della fase richiede prima un encoder IQ completo. La fase può iniziare con `Q2_K`/`Q3_K`, ma non può dirsi conclusa finché i target IQ non esistono e non sono verificati.
- La verifica “MSE/SNR migliorano” richiede le metriche della Fase 3 o un harness equivalente; non va dichiarata sulla sola presenza della matrice.
- Le fixture reali non sono incluse nel repository: vanno documentate per provenienza, licenza e hash.
- Evidenza aggiornata 2026-11-21: `ImportanceMatrix::load` legge legacy + GGUF, `type_overrides` seleziona i top-N per totale importanza, `transcode_with_imatrix` applica override utente prima del preset; `generate_with_llama_imatrix` richiama `llama-imatrix` esterno; `OVERFITTING_WARNING` pubblicata nel CLI.

---

## Fase 3 — Qualità misurabile (settimane 12–15)

**Stato:** ◐ parziale; metriche native, ranking, report, planner budget e KLD esterno presenti; mancano test 7B e confronto su 10 modelli.
**Aggiornamento:** il planner ora sceglie il target più piccolo quando nessun target entra nell'hardware e Q4_K_M quando l'hardware non è noto; prima poteva proporre F16 in entrambi i casi. Evidenza 2026-11-21: report CSV ora con escaping corretto per nomi tensore con virgole/virgolette/nuove righe; planner a budget preso in carico con prova di integrazione (test `quality_fit_and_ops_work_on_a_real_gguf`).
**Dipendenze:** Fase 2 per le metriche pesate con imatrix; Fase 4 per il diff fra GGUF.

### Obiettivi

- Calcolare metriche native per tensore: RMSE, SNR e similarità coseno; pesarle con l'imatrix quando presente.
- Produrre un ranking di sensibilità dei tensori.
- Quantizzazione a budget: dato un limite “X GB a Y contesto”, assegnare i bit per tensore.
- KLD opzionale richiamando `llama-perplexity`.
- Esportare un report JSON e CSV con i dati per tensore.

### Lavoro previsto

1. Decodifica dei tensori originali e quantizzati senza caricare l'intero modello in RAM, riusando il pipeline memory-bounded.
2. Metriche per tensore e aggregate, con esclusione esplicita dei tensori non quantizzabili.
3. Ranking normalizzato e spiegazione delle categorie (`attn_v`, `ffn_down`, output, embedding, resto).
4. Solver a budget con più tentativi di assegnazione, rispetto dei vincoli di block size e fallback.
5. Integrazione opzionale con `llama-perplexity` per KLD; nessuna inferenza reimplementata.
6. Schema JSON/CSV versionato, importabile e stabile per script.

### Criteri di accettazione

- Per un modello 7B, la quantizzazione a budget produce un file che entra nel limite richiesto con il minor degrado misurato possibile. → **Gate esterno** (modello 7B, Phase 8)
- L'SNR medio supera la soglia concordata e viene riportato insieme alla distribuzione per tensore.
- Il report JSON/CSV contiene una riga per tensore e i totali, ed è riproducibile a parità di input e parametri.
- Il ranking di sensibilità è stabile su una fixture ripetibile.

### Note

- “Minor degrado possibile” richiede una funzione obiettivo misurabile; la soglia SNR deve essere un parametro esplicito, non un numero implicito nel codice.
- La stima del fit usata dal budget deve essere condivisa con la Fase 5 per evitare due formule divergenti.

---

## Fase 4 — Operazioni GGUF (settimane 16–18)

**Stato:** ◐ parziale; split/merge con manifest, verifica SHA-256, comandi operativi, compilazione Jinja con minijinja, validazione rope/chat-template e rendering del template con un contesto JSON presenti; il controllo delle coppie rope contro il runtime effettivo resta esterno.
**Già presente:** parser GGUF, reblock PQ2_0 → Q2_0, split/merge con header originale, metadati, prune, diff, SHA-256/LFS, round-trip byte-identico verificato dal manifest, `meta --validate` con compilazione Jinja e `meta --render-template` con contesto JSON.
**Evidenza 2026-11-21:** `metadata_set` e `prune` ora rifiutano input == output (integrità: il file di origine resta intatto); test aggiunti.
**Assente:** controllo delle coppie rope contro il runtime effettivo.

### Obiettivi

- Split/merge shard nel formato `-00001-of-0000N`, con `--keep-split` e `--delete-splits`.
- Editor metadati completo: array, chat template con validazione, rope e altri KV.
- Prune layer e diff fra due GGUF.
- Verifica dell'integrità SHA-256 contro gli hash LFS di Hugging Face.

### Lavoro previsto

1. Modello di layout per shard con offset globali, allineamento e metadati `split.*`.
2. Scrittura di un GGUF con metadati modificati senza ricomprimere i tensori.
3. Merge che ricostruisce header, tabella tensori e dati nell'ordine originale.
4. Comandi `split`, `merge`, `meta`, `prune`, `diff`, `hash`.
5. Verifica hash LFS con gestione di file assenti, hash non disponibili e download parziali.
6. Validazione del chat template e dei tipi dei metadati prima della scrittura.

### Criteri di accettazione

- Merge di 3 shard: il file risultante è identico byte per byte all'originale se i tensori non vengono ricompressi. → verificato su shard reali (evidenza 2026-11-19, README).
- La modifica di un array di token non corrompe il GGUF e viene riletta correttamente.
- Uno split seguito da merge conserva offset, allineamento e tipi.
- Un hash LFS errato viene segnalato chiaramente e non viene confuso con un errore di conversione.

### Note

- Il merge byte-identico è possibile solo se l'ordine dei tensori, l'allineamento e i metadati sono preservati; il test deve fissare queste condizioni.
- La lettura dei file shard è anche un prerequisito per il download di modelli sharded nella Fase 6.

---

## Fase 5 — Stima fit RAM/VRAM (settimane 19–20)

**Stato:** ◐ parziale; pesi, KV per-layer GQA, sliding window, MLA e stato SSM presenti; confronto reale con llmvram/hf-mem e 70B non eseguito.
**Aggiornamento:** la GUI mostra un badge fit verde/giallo/rosso con dettaglio di pesi, KV cache e overhead; “Non so cosa scegliere” usa il planner a budget sul modello più grande in coda e mantiene il reblock lossless per PQ2_0. **Discover mostra un badge calcolato prima del download** dai byte pubblicati + 512 MiB di overhead: il KV cache non è calcolabile senza header, quindi è escluso e dichiarato nel tooltip, e con hardware sconosciuto il badge dice “Fit unknown”. Restano il confronto su 10 modelli, il caso 70B su 24 GB e il badge completo (con KV) nel dettaglio repo, che richiedono file reali.

### Obiettivi

- Calcolo esatto dei byte per tensore.
- KV cache corretta per GQA, finestre scorrevoli, attenzione ibrida e MLA.
- Badge “entra / non entra” nella vista Discover prima del download.
- Rilevamento automatico dell'hardware.

### Lavoro previsto

1. Calcolo dei byte dei pesi a partire dai tipi e dalle dimensioni GGUF, inclusi allineamenti e overhead.
2. Formula KV cache per layer, testa, dimensione, contesto, tipo cache e GQA; casi sliding-window/ibridi/MLA con fallback esplicito.
3. Overhead di runtime e backend configurabile, senza fingere precisione non disponibile.
4. Rilevamento CPU/RAM e GPU/VRAM per i backend supportati, con risultati “sconosciuto” quando non verificabili.
5. Badge e suggerimento nella GUI; stima disponibile anche via CLI/JSON. *(badge RAM/VRAM e planner a budget implementati)*
6. Confronto con `llmvram` e `hf-mem` su un set di modelli.

### Criteri di accettazione

- Confronto con tool esistenti su 10 modelli: errore inferiore al 5%. → **Gate esterno** (10 modelli reali)
- La stima per un modello 70B su GPU da 24 GB è coerente con la realtà. → **Gate esterno** (70B su 24 GB)
- Ogni stima riporta i parametri usati (contesto, cache, overhead) e non mostra un badge definitivo se l'hardware è sconosciuto.
- Il badge Discover viene calcolato prima del download. *(implementato sui soli pesi: il KV richiede l'header, assente prima dello scaricamento)*

### Note

- “Entra” dipende dal runtime, dal backend e dalla configurazione: la UI deve mostrare la base della stima, non solo il verdetto.
- I 10 modelli di confronto e le versioni dei tool esterni vanno registrati nel test.

---

## Fase 6 — Discover e installazione (settimane 21–24)

**Stato:** ◐ parziale  
**Già presente:** Discover, ricerca, repo detail, download parallelo 16 range dalla GUI e dalla CLI, resume `.part`, annullamento, validazione GGUF, mmproj CLI e GUI, hash LFS/SHA-256 dalla GUI, rilevamento dei file già presenti, indice cache con quota e poda selettiva (`cache` CLI e GUI), installazione LM Studio/Ollama e quick test, download headless con stream JSON-lines (`download --events`, `download-many --events`).  
**Evidenza 2026-11-21:** il probe HEAD del download ora ha un fallback a `Range: bytes=0-0` per i server che non rispondono al HEAD (test aggiunti).
**Assente:** installazione verificata su un ambiente Ollama/LM Studio reale, che richiede quelle applicazioni installate.

### Obiettivi

- Download parallelo con resume e verifica: fino a 16 connessioni per file e 8 file simultanei.
- Abbinamento automatico `mmproj` per modelli vision.
- Installazione in LM Studio/Ollama con Modelfile e template corretti.
- Test rapido opzionale con `llama-cli`/`llama-server`.
- Cache gestita e modalità headless con output JSON-lines.

### Lavoro previsto

1. Scheduler download con range requests, resume, limite di concorrenza e backoff.
2. Verifica hash LFS/SHA-256 e gestione dei file parziali.
3. Rilevamento mmproj e associazione al modello principale.
4. Installer LM Studio/Ollama tramite percorsi e CLI ufficiali, senza presumere che siano installati.
5. Test di caricamento opzionale con `llama-cli`/`llama-server`, timeout e output leggibile.
6. Cache directory, pulizia e comandi headless con eventi JSON-lines.

### Criteri di accettazione

- Installazione in Ollama: il modello appare e risponde a un prompt di test. → **Gate esterno** (applicazione installata)
- Installazione in LM Studio: il modello è importato e selezionabile. → **Gate esterno** (applicazione installata)
- Download interrotto e ripreso: il file finale ha hash corretto. (evidenza 2026-11-19, README)
- Se LM Studio/Ollama/llama.cpp non sono presenti, la funzione è disabilitata con un messaggio chiaro.

### Note

- Le installazioni toccano directory e processi esterni: servono conferma esplicita e nessuna modifica silenziosa.
- I test di installazione richiedono ambienti reali o mock documentati; non vanno dichiarati sulla sola compilazione.

---

## Fase 7 — Famiglia Prism (settimane 25–27)

**Stato:** ◐ parziale  
**Già presente:** reblock `PQ2_0 → Q2_0`, verifica lossless, test di split dei blocchi, rifiuto esplicito di PTQ1_0, rilevamento “già Q2_0”, metadati `prism.*` MTP/Hadamard preservati byte per byte; un file Prism 2B reale riconvertito è stato caricato in llama.cpp e ha prodotto lo stesso prefisso del runtime Prism.  
**Assente:** valutazione PTQ1_0, verifica numerica dell'esecuzione MTP/Hadamard, prova su 5 file reali.

### Obiettivi

- Consolidare il reblock `PQ2_0 → Q2_0` e la verifica bit-exact.
- Valutare `PTQ1_0 → Q2_0` (dense trits) verificando il layout prima di promettere losslessness.
- Rilevare automaticamente i file già compatibili con upstream ed evitare conversioni inutili.
- Gestire i metadati MTP/Hadamard nei file Bonsai 2.

### Criteri di accettazione

- Reblock su 5 file PQ2_0 reali: output caricabile su llama.cpp mainline e perplexity invariata. → **Gate esterno** (5 file reali)
- Un file già Q2_0 viene segnalato e non convertito automaticamente. (implementato nella GUI; evidenza codice)
- PTQ1_0 non viene dichiarato lossless finché il layout non è verificato e documentato.
- I metadati Hadamard/MTP rilevanti vengono preservati o esplicitamente rifiutati.

### Note

- Servono fixture reali e una build upstream; i test sintetici non bastano per l'accettazione.
- La verifica “stessi output/perplexity” richiede un runtime esterno e va trattata come test d'integrazione.

---

## Fase 8 — safetensors → GGUF (opzionale, tardiva)

**Stato:** ◐ parziale; implementazione completa nell'ambito dichiarato (bridge Python, dequant nativo FP8/AWQ → BF16, conversione streaming per LLaMA/Mistral/Mixtral/Qwen2/Qwen3 dense e MoE con BPE o SentencePiece) e verificata con checkpoint reali; manca un confronto numerico AWQ tensore-per-tensore. Altre architetture HF restano fuori e devono usare il bridge o fallire con un messaggio esplicito.
**Dipendenze:** decisione di prodotto, Python e `convert_hf_to_gguf.py` opzionali.

### Obiettivi

- Bridge opzionale verso `convert_hf_to_gguf.py` se presente.
- Implementazione nativa per `llama`, `mistral`, `mixtral`, `qwen2` e `qwen3` dense e MoE, con tokenizer BPE o SentencePiece.
- Importazione FP8/AWQ con dequantizzazione a BF16.

### Criteri di accettazione

- Se `convert_hf_to_gguf.py` è installato, il tool lo richiama e produce un GGUF valido.
- La dequantizzazione AWQ → BF16 è corretta con verifica numerica.
- Se Python o lo script non sono disponibili, la funzione è disabilitata con istruzioni chiare.
- Un checkpoint reale produce un GGUF apribile dal parser nativo, con tokenizer e metadati di architettura; verificato con TinyLlama, Qwen2.5 BF16/AWQ/FP8 e un Mixtral tiny, caricati da llama.cpp b12176.

### Note

- Il bridge non deve includere l'interprete Python né dipendenze del repository Hugging Face.
- La conversione nativa è un progetto separato e non deve bloccare le fasi 2–7.
- AWQ e FP8 sono stati provati su checkpoint reali: entrambi i GGUF si caricano e generano testo coerente. Il packing AWQ e i valori E4M3FN sono coperti da test numerici, ma non è stato conservato un confronto AWQ tensore-per-tensore.
- La conversione nativa copre LLaMA/Mistral/Mixtral/Qwen2/Qwen3 dense e MoE con BPE o SentencePiece; non copre tutte le architetture HF. I casi non supportati devono fallire esplicitamente o passare dal bridge.

---

## 5.2 — Guide in-app per newbie

**Stato:** ◐ parziale

### Già presente

- Descrizioni dei formati nella GUI e tooltip/preview delle scelte.
- Popover “?” su output format, destinazione e verifica, con spiegazione e comportamento consigliato.
- Modale di avviso per la requantizzazione con spiegazione del degrado.
- Messaggi di errore per file non supportati.
- Badge fit verde/giallo/rosso per RAM/VRAM, con tooltip che riporta pesi, KV cache, overhead e stato dei due device.
- Pulsante “Non so cosa scegliere” che usa il planner a budget sul modello più grande in coda; per i file PQ2_0 mantiene il reblock lossless.
- Avviso “VRAM: not detected” nel badge e voce “GPU not detected” nella guida offline; il suggerimento ripiega sulla RAM e lo registra nel log.
- **Evidenza 2026-11-21:** help modale aggiornato con voci nuove (cache Discover, badge download, “Non so cosa sceglire”); schermata vuota con i suggerimenti sulle scorciatoie Ctrl+O/Ctrl+S/Ctrl+L/F1.

### Da fare

- Test di usabilità con 5 utenti non esperti.

### Accettazione

- Un utente non esperto completa il flusso principale senza dover leggere il README.
- Ogni badge spiega la stima e non promette compatibilità non verificata.
- Le guide restano disponibili offline.

---

## 5.3 — Feedback visivo e prestazioni

**Stato:** ◐ parziale; log, progress e scorciatoie presenti; avvio misurato in 0,22–0,43 s su questa macchina; test di interazione con 5 utenti non eseguito.

### Già presente

- Barre di progresso per conversione e verifica, con velocità, ETA e tensor corrente.
- Lavoro lungo in worker separato; la UI resta reattiva.
- Download panel con byte, percentuale, velocità, ETA e annullamento.
- Log inferiore filtrabile, scorciatoie `Ctrl+O`, `Ctrl+S`, `Ctrl+L`, `F1` e avvio misurato sotto il secondo.

### Da fare

- Verifica delle scorciatoie con focus reale e test di interazione con 5 utenti.

### Accettazione

- La UI non si blocca durante un modello grande.
- I log non espongono token o path sensibili.
- Le scorciatoie funzionano con focus coerente.

---

## 5.4 — Criteri di accettazione UI

**Stato:** ◐ parziale; contrasto e avvio misurati, test con 5 utenti non eseguito.

- Test di usabilità con 5 utenti non esperti: flusso “scarica e installa un modello” completato in meno di 3 minuti senza aiuto (non eseguito).
- Contrasto colori conforme a WCAG AA (rapporto ≥ 4.5:1): verificato da test automatico su 12 coppie testo/stato.
- Avvio applicazione in meno di 1 secondo: misurato 0,22–0,43 s su questa macchina.

Questi criteri richiedono misurazioni e sessioni reali. Non vanno dichiarati soddisfatti sulla base della sola compilazione o di una verifica visiva.

---

## 6. Rischi e mitigazioni

| Rischio | Impatto | Mitigazione |
|---|---|---|
| Divergenza dai codec di llama.cpp | Alto | Test differenziali in CI con una build pinnata |
| Bug upstream che si riflettono su RustBlock | Medio | Override espliciti prioritari e test di regressione |
| Architetture esotiche non gestite | Medio | Rilevamento, messaggio chiaro e fallback a “non quantizzare” |
| Download lenti | Basso | Download parallelo con resume |
| Complessità GUI per utenti esperti | Basso | Modalità avanzata separata e CLI speculare |
| Errori nella scrittura GGUF | Alto | Scrittura temporanea, verifica, rename atomico e test di round-trip |

## 7. Metriche di successo

| Area | Metrica |
|---|---|
| Correttezza | 100% di match bit-identico con `llama-quantize` sui tipi deterministici |
| Qualità | SNR medio della quantizzazione a budget superiore del 10% rispetto a `Q4_K_M` standard |
| Usabilità | 90% dei nuovi utenti completa il flusso base senza errori |
| Prestazioni | Download di un modello 7B in meno di 2 minuti su 100 Mbps |

Il match bit-identico va misurato con una build di riferimento fissata, non dedotto dalla somiglianza delle ricette. La metrica SNR richiede la Fase 3 e una soglia esplicita.

---

## 8. Ordine di lavoro e gate

0. **Aggiornamento del 21 novembre 2026 — revisione a fondo (fatto):** revisione del progetto, revisione della roadmap, implementazioni di miglioramento, test, fmt e clippy. Vedi “Aggiornamento del 21 novembre 2026” in cima.
1. **Gate correttezza:** ✅ build di riferimento congelata a llama.cpp **b12359** (commit b1b770d90, asset `llama-b12359-bin-ubuntu-x64.tar.gz`) e test differenziale eseguito con esito positivo il 2026-11-19. Il job CI `parity` in `.github/workflows/ci.yml` scarica la stessa release e ripete il confronto. Resta fuori solo la parità byte-per-byte dell'output.
2. **Fase 2:** parser imatrix, pesi per `Q2_K`/`Q3_K`, generazione esterna opzionale; la parte IQ resta bloccata finché non esiste un encoder IQ.
3. **Fase 4:** split/merge e metadati, perché sbloccano shard, hash e parte della Fase 6.
4. **Fase 3:** metriche e budget, riusando la stima fit della Fase 5.
5. **Fase 5:** fit RAM/VRAM e badge Discover.
6. **Fase 6:** download parallelo, mmproj, installer e headless.
7. **Fase 7:** validazione Prism su file reali.
8. **Fase 8:** bridge safetensors opzionale.

Le fasi possono essere accorpate o riordinate solo se i gate di correttezza restano invariati. Ogni fase deve aggiornare questo documento con evidenze, test eseguiti e limiti residui.

## 9. Definition of done

Una fase è conclusa solo quando:

- i criteri di accettazione sono verificati con test o prove documentate;
- la CLI e la GUI hanno lo stesso comportamento, o la differenza è documentata;
- gli errori esterni (runtime, Python, LM Studio, Ollama) sono gestiti con messaggi chiari;
- non ci sono regressioni nei test esistenti;
- README e questo documento sono aggiornati;
- i limiti non verificati sono indicati esplicitamente.
- **Evidenza applicabile oggi:** `cargo test --workspace` passa (21 nov 2026), `cargo fmt --all -- --check` e `cargo clippy --workspace --all-targets -- -D warnings` passano; ogni modifica successiva al 21 nov 2026 aggiunge la verifica eseguita e i nuovi test e mantiene questo criterio.

## 10. Fixture e segreti

- Le fixture reali devono essere documentate con provenienza, licenza e hash; i file di modello non vanno committati.
- I token Hugging Face restano in variabili d'ambiente e non vengono scritti nei log o nei report.
- Gli strumenti esterni (`llama-imatrix`, `llama-perplexity`, `llama-cli`, `llama-server`, Python) sono opzionali e vengono rilevati, non scaricati automaticamente.
- Nessuna fase deve richiedere l'accesso a credenziali o servizi non documentati.

## 11. Priorità e raccomandazione

La correttezza del motore resta il prerequisito: senza parità verificabile con una build di riferimento di `llama-quantize`, le funzioni di qualità e budget non sono credibili. La Fase 1 è completata; il prossimo lavoro deve partire dal gate di correttezza e dalle fasi che sbloccano più dipendenze (imatrix e operazioni GGUF).

La GUI può diventare il fattore decisivo per l'adozione, ma solo dopo che il motore è solido. Le guide in-app e i badge di compatibilità non devono anticipare stime che la Fase 5 non è ancora in grado di verificare.

Procedimenti per chiudere i gate aperti: usare il codice prodotto per produrre modelli di test durante le valutazioni esterne (7B/10 modelli/70B), aggiungere il confronto seguente, eseguire Ollama/LM Studio su una macchina dedicata e documentare i risultati; poi rilasciare una revisione del 2026-12-XX.
