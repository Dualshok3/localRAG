# Eval: Suchmodi im Vergleich

Korpus: 16 synthetische Dokumente · 40 Fragen · Embedding-Modell: BAAI/bge-m3 (cuda) · Indexierung inkl. Vektoren: 0.8 s

| Modus | Fragen | n | Recall@1 | Recall@3 | Recall@5 | MRR | Latenz p50 |
|---|---|---|---|---|---|---|---|
| **keyword** | gesamt | 40 | 0.62 | 0.72 | 0.72 | 0.67 | 0.4 ms |
|  | disambiguation | 10 | 0.50 | 0.90 | 0.90 | 0.67 | 0.5 ms |
|  | exact-id | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.4 ms |
|  | keyword | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 0.3 ms |
|  | paraphrase | 13 | 0.23 | 0.23 | 0.23 | 0.23 | 0.3 ms |
| **semantic** | gesamt | 40 | 0.88 | 1.00 | 1.00 | 0.93 | 27.6 ms |
|  | disambiguation | 10 | 0.90 | 1.00 | 1.00 | 0.95 | 26.8 ms |
|  | exact-id | 5 | 0.40 | 1.00 | 1.00 | 0.67 | 27.3 ms |
|  | keyword | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 28.2 ms |
|  | paraphrase | 13 | 0.92 | 1.00 | 1.00 | 0.96 | 28.5 ms |
| **hybrid** | gesamt | 40 | 0.82 | 1.00 | 1.00 | 0.90 | 27.6 ms |
|  | disambiguation | 10 | 0.70 | 1.00 | 1.00 | 0.83 | 27.6 ms |
|  | exact-id | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 27.5 ms |
|  | keyword | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 28.2 ms |
|  | paraphrase | 13 | 0.69 | 1.00 | 1.00 | 0.83 | 27.5 ms |

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
| paraphrase | Darf ich eine Katze halten? | mietvertrag.md | – | 1 | 2 |
| paraphrase | Was hat mich Elektrizität letztes Jahr gekostet? | stromrechnung_2024.md | – | 1 | 2 |
| paraphrase | Wie viel muss ich bei einem Unfall mit dem Auto selbst bezahlen? | kfz_versicherung.md | – | 1 | 1 |
| paraphrase | Wie viele freie Tage bekomme ich im Job? | arbeitsvertrag.md | – | 2 | 3 |
| paraphrase | Bis wann kann ich mein Notebook reklamieren? | laptop_kaufbeleg.md | – | 1 | 1 |
| paraphrase | Wo schlafen wir während der Sommerferien? | italien_reise.md | – | 1 | 2 |
| paraphrase | Welches überbackene Nudelgericht hatte ich aufgeschrieben? | lasagne_rezept.md | – | 1 | 1 |
| paraphrase | Wie setze ich das WLAN-Gerät auf Werkseinstellungen zurück? | router_manual.md | – | 1 | 1 |
| paraphrase | Wie viel Internet habe ich auf dem Smartphone? | handyvertrag.md | – | 1 | 1 |
| exact-id | PF-TEST-1234 | laptop_kaufbeleg.md | 1 | 1 | 1 |
| exact-id | 1ESY-0000-TEST | stromrechnung_2024.md | 1 | 2 | 1 |
| exact-id | 192.168.0.1 | router_manual.md | 1 | 1 | 1 |
| exact-id | 2.306 EUR | nebenkostenabrechnung_2023.md | 1 | 3 | 1 |
| disambiguation | Wie hoch war der Stromverbrauch im Jahr 2023? | stromrechnung_2023.md | 3 | 1 | 1 |
| disambiguation | Wie lange ist die Kündigungsfrist für die Garage? | garagenmietvertrag.md | 3 | 1 | 2 |
| disambiguation | Wie viele Urlaubstage hatte ich beim alten Arbeitgeber? | alter_arbeitsvertrag_2019.md | 1 | 1 | 1 |
| disambiguation | Wie viel verdiene ich aktuell brutto? | arbeitsvertrag.md | – | 1 | 3 |
| paraphrase | Ist mein Rad versichert, wenn es geklaut wird? | hausratversicherung.md | 1 | 1 | 1 |
| paraphrase | Wann war die erste Inspektion vom Rad fällig? | fahrrad_kaufbeleg.md | 1 | 1 | 1 |
| keyword | Was zahlt die Krankenkasse bei der Zahnreinigung dazu? | zahnarzt_rechnung.md | 1 | 1 | 1 |
| exact-id | WTEST-998877 | fahrrad_kaufbeleg.md | 1 | 2 | 1 |
| paraphrase | Darf ich Benzinkanister in der Garage lagern? | garagenmietvertrag.md | 1 | 1 | 1 |
| disambiguation | Gab es 2023 eine Rückzahlung vom Stromanbieter? | stromrechnung_2023.md | 2 | 1 | 1 |
| disambiguation | Wie hoch war der Grundpreis beim Strom 2023? | stromrechnung_2023.md | 1 | 1 | 1 |
| disambiguation | Arbeitspreis Strom 2024 | stromrechnung_2024.md | 1 | 1 | 1 |
| disambiguation | Welcher Abschlag galt ab März 2024? | stromrechnung_2023.md | 2 | 2 | 2 |
| disambiguation | Was stand im Arbeitsvertrag von 2019 zur Arbeitszeit? | alter_arbeitsvertrag_2019.md | 1 | 1 | 1 |
| disambiguation | Wie viel habe ich 2024 für Strom bezahlt? | stromrechnung_2024.md | 1 | 1 | 1 |
