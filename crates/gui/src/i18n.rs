//! Small, dependency-free translation table (English / Italian).

use crate::model::Lang;

pub struct T {
    pub tagline: &'static str,
    pub subtitle: &'static str,
    pub about: &'static str,
    pub close: &'static str,
    pub language: &'static str,
    pub footer_lossless: &'static str,
    pub footer_offline: &'static str,
    pub footer_license: &'static str,

    pub drop_title: &'static str,
    pub drop_hint: &'static str,
    pub browse: &'static str,
    pub drop_overlay: &'static str,
    pub drop_only_gguf: &'static str,

    pub queue: &'static str,
    pub queue_empty: &'static str,
    pub clear_finished: &'static str,
    pub clear_all: &'static str,
    pub remove: &'static str,
    pub open_folder: &'static str,

    pub options: &'static str,
    pub verification: &'static str,
    pub verify_none: &'static str,
    pub verify_structure: &'static str,
    pub verify_full: &'static str,
    pub verify_help_none: &'static str,
    pub verify_help_structure: &'static str,
    pub verify_help_full: &'static str,
    pub verify_only: &'static str,
    pub verify_only_help: &'static str,
    pub overwrite: &'static str,
    pub output_mode: &'static str,
    pub output_next: &'static str,
    pub output_custom: &'static str,
    pub choose_folder: &'static str,
    pub no_folder: &'static str,
    pub output_exists: &'static str,

    pub convert_n: &'static str,
    pub verify_n: &'static str,
    pub processing: &'static str,
    pub stop: &'static str,
    pub stop_hint: &'static str,
    pub log_nothing: &'static str,

    pub status_ready: &'static str,
    pub status_done: &'static str,
    pub status_verified: &'static str,
    pub status_failed: &'static str,
    pub status_native: &'static str,
    pub status_ptq1: &'static str,
    pub status_other: &'static str,
    pub status_invalid: &'static str,
    pub phase_converting: &'static str,
    pub phase_verifying: &'static str,

    pub size: &'static str,
    pub output_estimate: &'static str,
    pub tensors: &'static str,
    pub pq2_tensors: &'static str,
    pub hadamard: &'static str,
    pub output: &'static str,
    pub no_output_needed: &'static str,

    pub overall: &'static str,
    pub speed: &'static str,
    pub eta: &'static str,
    pub elapsed: &'static str,
    pub file_of: &'static str,

    pub log: &'static str,
    pub export_log: &'static str,
    pub clear_log: &'static str,
    pub log_ready: &'static str,
    pub log_added: &'static str,
    pub log_duplicate: &'static str,
    pub log_analysis_failed: &'static str,
    pub log_start_convert: &'static str,
    pub log_start_verify: &'static str,
    pub log_done: &'static str,
    pub log_verified: &'static str,
    pub log_failed: &'static str,
    pub log_batch_convert: &'static str,
    pub log_batch_verify: &'static str,
    pub log_saved: &'static str,
    pub log_save_failed: &'static str,
    pub log_output_conflict: &'static str,
    pub log_worker_lost: &'static str,
    pub log_stopping: &'static str,
    pub log_stopped: &'static str,
    pub log_close_blocked: &'static str,

    pub about_text: &'static str,
    pub credits: &'static str,
    pub version: &'static str,

    pub verified_pq2: &'static str,
    pub verified_native: &'static str,
    pub verified_other: &'static str,
    pub verified_ptq1: &'static str,

    pub files_n: &'static str,
    pub est_output_total: &'static str,
}

pub const EN: T = T {
    tagline: "PQ2_0 -> Q2_0, with zero quality loss",
    subtitle: "For Prism ML Ternary Bonsai models - offline and private",
    about: "About",
    close: "Close",
    language: "Language",
    footer_lossless: "Lossless",
    footer_offline: "Offline",
    footer_license: "Apache-2.0",

    drop_title: "Drop GGUF files here",
    drop_hint: "or browse your computer - multiple files are supported",
    browse: "Browse files…",
    drop_overlay: "Release to add",
    drop_only_gguf: "Only .gguf files are supported",

    queue: "Queue",
    queue_empty: "No files yet. Drop a GGUF above to get started.",
    clear_finished: "Remove finished",
    clear_all: "Clear all",
    remove: "Remove",
    open_folder: "Open folder",

    options: "Options",
    verification: "Verification",
    verify_none: "None (fastest)",
    verify_structure: "Structure (recommended)",
    verify_full: "Full (bit-exact)",
    verify_help_none:
        "Writes the output without re-reading it. Fastest, but cannot catch write errors.",
    verify_help_structure:
        "Checks header, offsets, alignment and file size. Takes a few seconds.",
    verify_help_full:
        "Re-reads every value and compares it bit-for-bit. Roughly doubles I/O time.",
    verify_only: "Verify only (write nothing)",
    verify_only_help:
        "Checks files without producing output. PQ2_0 files get the full lossless check.",
    overwrite: "Overwrite existing outputs",
    output_mode: "Output folder",
    output_next: "Next to each input file",
    output_custom: "Custom folder",
    choose_folder: "Choose…",
    no_folder: "No folder selected",
    output_exists: "Output already exists (enable “Overwrite”)",

    convert_n: "Convert {} file(s)",
    verify_n: "Verify {} file(s)",
    processing: "Processing…",
    stop: "Stop after this file",
    stop_hint: "Finishes the current file, then cancels the remaining queue.",
    log_nothing: "Nothing to process.",

    status_ready: "Ready",
    status_done: "Done",
    status_verified: "Verified",
    status_failed: "Error",
    status_native: "Already Q2_0",
    status_ptq1: "PTQ1_0 not supported",
    status_other: "Not a ternary model",
    status_invalid: "Invalid file",
    phase_converting: "Converting",
    phase_verifying: "Verifying",

    size: "Size",
    output_estimate: "Estimated output",
    tensors: "tensors",
    pq2_tensors: "PQ2_0 tensors",
    hadamard: "Hadamard metadata",
    output: "Output",
    no_output_needed: "No output needed",

    overall: "Overall progress",
    speed: "Speed",
    eta: "ETA",
    elapsed: "Elapsed",
    file_of: "File {} of {}",

    log: "Log",
    export_log: "Export…",
    clear_log: "Clear",
    log_ready: "Ready. Drop a GGUF file to begin.",
    log_added: "Added: {}",
    log_duplicate: "Already queued: {}",
    log_analysis_failed: "Cannot read {}: {}",
    log_start_convert: "Starting conversion of {} file(s)…",
    log_start_verify: "Starting verification of {} file(s)…",
    log_done: "Converted {} ({} -> {}) in {}",
    log_verified: "Verified {} in {}",
    log_failed: "Error on {}: {}",
    log_batch_convert: "Conversion finished: {} succeeded, {} failed, {}",
    log_batch_verify: "Verification finished: {} verified, {} failed, {}",
    log_saved: "Log saved to {}",
    log_save_failed: "Could not save the log: {}",
    log_output_conflict: "Duplicate output path for {}: {}",
    log_worker_lost: "The background worker stopped unexpectedly.",
    log_stopping: "Stopping after the current file…",
    log_stopped: "Batch stopped. Remaining files are ready to start again.",
    log_close_blocked: "A conversion is running. Stop it before closing the window.",

    about_text:
        "Reblock converts Ternary Bonsai GGUFs from PQ2_0 (Prism’s private layout) to standard Q2_0 without decoding or requantizing. The source file is never modified.",
    credits:
        "Apache-2.0. Not affiliated with Prism ML or Alibaba Cloud. “Bonsai” is a Prism ML product; Qwen3.8-27B is by Alibaba Cloud.",
    version: "Version",

    verified_pq2: "Lossless split verified - {} PQ2_0 tensors",
    verified_native: "Structure OK - {} tensors, already Q2_0",
    verified_other: "Structure OK - {} tensors",
    verified_ptq1: "Structure OK - PTQ1_0 cannot be converted losslessly",

    files_n: "{} file(s)",
    est_output_total: "Output ~ {}",
};

pub const IT: T = T {
    tagline: "PQ2_0 -> Q2_0, senza perdita di qualità",
    subtitle: "Per i modelli Ternary Bonsai di Prism ML - offline e privato",
    about: "Informazioni",
    close: "Chiudi",
    language: "Lingua",
    footer_lossless: "Senza perdita",
    footer_offline: "Offline",
    footer_license: "Apache-2.0",

    drop_title: "Trascina qui i file GGUF",
    drop_hint: "oppure sfoglia il computer - puoi aggiungere più file",
    browse: "Sfoglia file…",
    drop_overlay: "Rilascia per aggiungere",
    drop_only_gguf: "Sono supportati solo file .gguf",

    queue: "Coda",
    queue_empty: "Nessun file in coda. Trascina un GGUF qui sopra.",
    clear_finished: "Rimuovi completati",
    clear_all: "Svuota",
    remove: "Rimuovi",
    open_folder: "Apri cartella",

    options: "Opzioni",
    verification: "Verifica",
    verify_none: "Nessuna (più veloce)",
    verify_structure: "Struttura (consigliata)",
    verify_full: "Completa (bit-exact)",
    verify_help_none:
        "Scrive il file senza rileggerlo. È la più veloce, ma non rileva errori di scrittura.",
    verify_help_structure:
        "Controlla header, offset, allineamento e dimensione. Richiede pochi secondi.",
    verify_help_full:
        "Rilegge ogni valore e lo confronta bit per bit. Raddoppia il tempo di I/O.",
    verify_only: "Solo verifica (non scrive nulla)",
    verify_only_help:
        "Controlla i file senza creare output. Per i PQ2_0 esegue la verifica completa.",
    overwrite: "Sovrascrivi gli output esistenti",
    output_mode: "Cartella di output",
    output_next: "Accanto a ogni file",
    output_custom: "Cartella personalizzata",
    choose_folder: "Scegli…",
    no_folder: "Nessuna cartella selezionata",
    output_exists: "Il file di output esiste già (attiva \"Sovrascrivi\")",

    convert_n: "Converti {} file",
    verify_n: "Verifica {} file",
    processing: "In corso…",
    stop: "Ferma dopo questo file",
    stop_hint: "Completa il file corrente e annulla i successivi.",
    log_nothing: "Nessun file da elaborare.",

    status_ready: "Pronto",
    status_done: "Completato",
    status_verified: "Verificato",
    status_failed: "Errore",
    status_native: "Già Q2_0",
    status_ptq1: "PTQ1_0 non supportato",
    status_other: "GGUF non ternario",
    status_invalid: "File non valido",
    phase_converting: "Conversione",
    phase_verifying: "Verifica",

    size: "Dimensione",
    output_estimate: "Output stimato",
    tensors: "tensori",
    pq2_tensors: "tensori PQ2_0",
    hadamard: "Metadati Hadamard",
    output: "Output",
    no_output_needed: "Nessun output necessario",

    overall: "Avanzamento totale",
    speed: "Velocità",
    eta: "Tempo rimanente",
    elapsed: "Trascorso",
    file_of: "File {} di {}",

    log: "Registro",
    export_log: "Esporta…",
    clear_log: "Pulisci",
    log_ready: "Pronto. Trascina un GGUF per iniziare.",
    log_added: "Aggiunto: {}",
    log_duplicate: "Già in coda: {}",
    log_analysis_failed: "Impossibile leggere {}: {}",
    log_start_convert: "Avvio conversione di {} file…",
    log_start_verify: "Avvio verifica di {} file…",
    log_done: "Convertito {} ({} -> {}) in {}",
    log_verified: "Verificato {} in {}",
    log_failed: "Errore su {}: {}",
    log_batch_convert: "Conversione terminata: {} riusciti, {} errori, {}",
    log_batch_verify: "Verifica terminata: {} verificati, {} errori, {}",
    log_saved: "Registro salvato in {}",
    log_save_failed: "Salvataggio registro non riuscito: {}",
    log_output_conflict: "Output duplicato per {}: {}",
    log_worker_lost: "Il processo in background si è interrotto in modo imprevisto.",
    log_stopping: "Arresto dopo il file corrente…",
    log_stopped: "Batch interrotto. I file rimanenti sono pronti per ripartire.",
    log_close_blocked: "Una conversione è in corso. Fermala prima di chiudere la finestra.",

    about_text:
        "Reblock converte i GGUF Ternary Bonsai da PQ2_0 (formato proprietario Prism) a Q2_0 standard, senza decodifica né riquantizzazione. Il file originale non viene mai modificato.",
    credits:
        "Licenza Apache-2.0. Non affiliato a Prism ML o Alibaba Cloud. \"Bonsai\" è un prodotto di Prism ML; Qwen3.8-27B è di Alibaba Cloud.",
    version: "Versione",

    verified_pq2: "Divisione lossless verificata - {} tensori PQ2_0",
    verified_native: "Struttura OK - {} tensori, già Q2_0",
    verified_other: "Struttura OK - {} tensori",
    verified_ptq1: "Struttura OK - PTQ1_0 non è convertibile senza perdita",

    files_n: "{} file",
    est_output_total: "Output ~ {}",
};

pub fn t(lang: Lang) -> &'static T {
    match lang {
        Lang::En => &EN,
        Lang::It => &IT,
    }
}

/// Replaces each `{}` in `template` with the next displayable argument.
/// Used for translated strings, where `format!` cannot take a runtime
/// template.
pub fn fill(template: &str, args: &[&dyn std::fmt::Display]) -> String {
    let mut out = String::with_capacity(template.len() + 32);
    let mut rest = template;
    let mut args = args.iter();

    while let Some(pos) = rest.find("{}") {
        out.push_str(&rest[..pos]);
        match args.next() {
            Some(arg) => out.push_str(&arg.to_string()),
            None => out.push_str("{}"),
        }
        rest = &rest[pos + 2..];
    }
    out.push_str(rest);
    out
}
