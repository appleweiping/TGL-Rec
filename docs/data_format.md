# Data Format

Interaction rows use:

```json
{"user_id":"u1","item_id":"i1","timestamp":1,"rating":5.0,"domain":"movies"}
```

Item rows use:

```json
{"item_id":"i1","title":"Item","description":null,"category":"A","brand":null,"domain":"movies","raw_text":"Item A"}
```

Prediction rows must follow the shared schema:

```json
{"user_id":"u1","target_item":"i2","candidate_items":["i1","i2"],"predicted_items":["i2","i1"],"scores":[1.0,0.0],"method":"method","domain":"movies","raw_output":null,"metadata":{}}
```

Reportable comparisons must not mix split or candidate protocols.
