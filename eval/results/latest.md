# Eval: Suchmodi im Vergleich

Korpus: 10 synthetische Dokumente · 25 Fragen · Embedding-Modell: BAAI/bge-m3 (cuda) · Indexierung inkl. Vektoren: 0.4 s

| Modus | Fragen | n | Recall@1 | Recall@3 | Recall@5 | MRR | Latenz p50 |
|---|---|---|---|---|---|---|---|
| **keyword** | gesamt | 25 | 0.60 | 0.60 | 0.60 | 0.60 | 0.3 ms |
|  | exact-id | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 0.3 ms |
|  | keyword | 11 | 1.00 | 1.00 | 1.00 | 1.00 | 0.3 ms |
|  | paraphrase | 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.3 ms |
| **semantic** | gesamt | 25 | 0.92 | 1.00 | 1.00 | 0.96 | 24.3 ms |
|  | exact-id | 4 | 0.50 | 1.00 | 1.00 | 0.75 | 28.9 ms |
|  | keyword | 11 | 1.00 | 1.00 | 1.00 | 1.00 | 24.2 ms |
|  | paraphrase | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 24.3 ms |
| **hybrid** | gesamt | 25 | 0.96 | 1.00 | 1.00 | 0.98 | 26.2 ms |
|  | exact-id | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 26.1 ms |
|  | keyword | 11 | 1.00 | 1.00 | 1.00 | 1.00 | 25.3 ms |
|  | paraphrase | 10 | 0.90 | 1.00 | 1.00 | 0.95 | 29.2 ms |

## Rang der richtigen Datei pro Frage

| Typ | Frage | Erwartet | keyword | semantic | hybrid |
|---|---|---|---|---|---|
| keyword | Kündigungsfrist Mietvertrag | mietvertrag.md | 1 | 1 | 1 |
| keyword | Kaution Wohnung | mietvertrag.md | 1 | 1 | 1 |
| keyword | Stromverbrauch kWh 2024 | stromrechnung_2024.md | 1 | 1 | 1 |
| keyword | Selbstbeteiligung Vollkasko | kfz_versicherung.md | 1 | 1 | 1 |
| keyword | Urlaubstage Arbeitsvertrag | arbeitsvertrag.md | 1 | 1 | 1 |
| keyword | Garantie Laptop Seriennummer | laptop_kaufbeleg.md | 1 | 1 | 1 |
| keyword | Hotel Florenz | italien_reise.md | 1 | 1 | 1 |
| keyword | Lasagne Béchamel | lasagne_rezept.md | 1 | 1 | 1 |
| keyword | router reset button | router_manual.md | 1 | 1 | 1 |
| keyword | Datenvolumen Handyvertrag | handyvertrag.md | 1 | 1 | 1 |
| keyword | Nachzahlung Nebenkosten 2023 | nebenkostenabrechnung_2023.md | 1 | 1 | 1 |
| paraphrase | Wie lange vorher muss ich Bescheid sagen, wenn ich ausziehen will? | mietvertrag.md | – | 1 | 1 |
| paraphrase | Darf ich eine Katze halten? | mietvertrag.md | – | 1 | 1 |
| paraphrase | Was hat mich Elektrizität letztes Jahr gekostet? | stromrechnung_2024.md | – | 1 | 1 |
| paraphrase | Wie viel muss ich bei einem Unfall mit dem Auto selbst bezahlen? | kfz_versicherung.md | – | 1 | 1 |
| paraphrase | Wie viele freie Tage bekomme ich im Job? | arbeitsvertrag.md | – | 1 | 1 |
| paraphrase | Bis wann kann ich mein Notebook reklamieren? | laptop_kaufbeleg.md | – | 1 | 1 |
| paraphrase | Wo schlafen wir während der Sommerferien? | italien_reise.md | – | 1 | 2 |
| paraphrase | Welches überbackene Nudelgericht hatte ich aufgeschrieben? | lasagne_rezept.md | – | 1 | 1 |
| paraphrase | Wie setze ich das WLAN-Gerät auf Werkseinstellungen zurück? | router_manual.md | – | 1 | 1 |
| paraphrase | Wie viel Internet habe ich auf dem Smartphone? | handyvertrag.md | – | 1 | 1 |
| exact-id | PF-TEST-1234 | laptop_kaufbeleg.md | 1 | 1 | 1 |
| exact-id | 1ESY-0000-TEST | stromrechnung_2024.md | 1 | 2 | 1 |
| exact-id | 192.168.0.1 | router_manual.md | 1 | 1 | 1 |
| exact-id | 2.306 EUR | nebenkostenabrechnung_2023.md | 1 | 2 | 1 |
