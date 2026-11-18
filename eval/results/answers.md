# Antwortqualität

Erzeugt von `eval/run_answer_eval.py`. Prüfmodell: `run-20261109T095346Z-1b210ccf/candidate-0014`. Fragen: `eval/answers.jsonl` (erfundener Korpus). Suche: hybrid, dieselben Quellen für alle Modelle.

| Modell | richtig | belegt | Zitat auf richtige Datei | „weiß ich nicht“ korrekt | fälschlich verweigert | Antwortzeit p50 |
|---|---|---|---|---|---|---|
| gemma-4-26b-a4b-it | 20/20 | 20/20 | 20/20 | 5/5 | 0/20 | 2.8 s |
| qwen3.5-9b | 17/20 | 19/20 | 20/20 | 4/5 | 0/20 | 2.9 s |
| run-20261109T095346Z-1b210ccf/candidate-0014 | 18/20 | 20/20 | 19/20 | 5/5 | 0/20 | 6.8 s |

## Falsch oder unbelegt

- **qwen3.5-9b** · Darf ich eine Katze halten? → falsch: 'Ja, du darfst eine Katze halten, da Katzen als Kleintiere gelten und die Haltung von Kleintieren in Käfigen ausgenommen ist [2]. Die vorherige Zustimmung des Ve' (Die Antwort widerspricht der Kernaussage, da sie behauptet, Katzen seien Kleintiere in Käfigen und bräuchten keine Zustimmung, während die Quelle explizit besag)
- **qwen3.5-9b** · Musste ich für 2023 beim Strom nachzahlen? → falsch: 'Ja, du musstest für das Jahr 2023 beim Strom nachzahlen. Der Gesamtbetrag betrug 1.098,48 EUR, während die geleisteten Abschläge 1.140,00 EUR betrugen, was zu e' (Die Antwort ist in sich widersprüchlich: Sie beginnt mit 'Ja, du musstest nachzahlen', stellt dann aber im selben Satz und im Folgesatz klar, dass ein Guthaben )
- **qwen3.5-9b** · Bis wann habe ich Garantie auf meinen Laptop? → falsch: 'Du hast eine Herstellergarantie von 24 Monaten ab dem Kaufdatum des Laptops am 12.03.2024 [1]. Das bedeutet, die Garantie gilt bis zum **12.09.2025**.' (Die Antwort enthält ein falsches Enddatum (12.09.2025 statt März 2026), wodurch die Kernaussage nicht erfüllt wird.)
- **qwen3.5-9b** · Welche Schadenfreiheitsklasse habe ich in der Teilkasko? → antwortet statt „weiß ich nicht“: 'In deiner Teilkasko hast du die Schadenfreiheitsklasse SF 5 [1].'
- **run-20261109T095346Z-1b210ccf/candidate-0014** · Wie hoch ist mein aktueller monatlicher Stromabschlag? → falsch: 'Dein neuer monatlicher Abschlag ab März 2024 beträgt 65,00 EUR [2].' (Die Antwort enthält nicht die geforderte Kernaussage (70 EUR), sondern gibt einen veralteten Wert aus Quelle [2] wieder. Zudem bezieht sie sich auf das falsche )
- **run-20261109T095346Z-1b210ccf/candidate-0014** · Bis wann habe ich Garantie auf meinen Laptop? → falsch: 'Die Herstellergarantie für deinen Laptop beträgt 24 Monate ab Kaufdatum [1].' (Die Kernaussage verlangt zwingend die Nennung des Kaufdatums (12.03.2024), um die Information vollständig zu machen. Da das Kaufdatum in der Antwort fehlt, ist )
