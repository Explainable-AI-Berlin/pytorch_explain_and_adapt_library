"""PEAL web demo: upload an ONNX classifier and a zipped image-folder dataset,
get the DiDAE Clever Hans analysis and a corrected classifier back.

    uvicorn peal.web.app:app --host 0.0.0.0 --port 8080

See peal/web/app.py for the HTTP API and deploy/spark for the DGX Spark setup.
"""
