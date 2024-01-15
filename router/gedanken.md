# Ziele
- Earliest Arrival sehr wichtig
- Alle Verbindungen anzeigen
- Einfache Verwendung -> Nur nötiges anzeigen
- Alternativen bei Suche anzeigen
- Zuverlässigste Route wählen
- Von Zug aus
- Nahverkehrsfreundlich



# Fragen?
- Muss es wirklich auch Nahverkehrsverbidungen anzeigen? -> Ja
- Wie viele Stunden in die Zukunft kann ich suchen, bis das System überlastet
- Für welchen Zeitraum (Zukunft, Vergangenheit) soll das routing funktionieren



# Kriterien zur Verbindungswahl
- Schnelligkeit
- Earliest arrival time
- Bequemlichkeit - vielleicht leere Züge / Nebenbahnen
- Preis - irrelevant bei Nahverkehr
- Streckenschönheit
- Passende Umsteigezeit (10 min) mehr ist zu viel, weniger zu wenig
- Umsteigezeiten


# Sinnvolle Pareto Kriterien
- Earliest arrival time
- Shortest Duration
- Transfers
- No long distance train
-> Only compare if one journey completely overlaps another
Complete Overlap:
=====
=====

====
 =
...

No Complete overlap
====
  ====

====
     ====

## Which connection is best?
- Earliest arrival
- All cons with fewest transfer
- duration not over + n % of minimal duration


# Was nicht sinnvoll ist
- Mehrfach am gleichen Bahnhof halten
- Aus einem Zug austeigen, und später wieder in ihn einsteigen
- Aus einem Zug aussteigen, und in einen mit der gleichen Route einsteige
- Nicht in einen Zug einsteigen, in den man schon früher hätte einsteigen können
- Deutlich vom Ziel wegfahren


# Router hyperparameters
- Max transfers ~ 10
- Max duration depending on the minimal duration
- Max departure window ~ 3h
- Filters (Bike, ICE, ...)
- Min transfer time
- Max transfer time


# Was ist ein Umweg?
- Unnnötig viel Fahrzeit
- Zu viele Umstieg


# Was ist kein Umweg
- Nahverkehr benutzen
- Eine andere route fahren, die kein krasser Umweg ist