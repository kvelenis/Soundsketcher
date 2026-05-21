# SoundSketcher Deployment Notes

This deployment keeps the legacy app alive and adds the refactor as a separate
auto-restarting service.

## Target Layout

- Legacy app:
  - source: `/mnt/ssd1/kvelenis/soundsketcher`
  - process: `main_with_event_detection.py`
  - internal port: `5002`
  - public URL: `https://helen.mus.auth.gr/app1/`
- Refactor app:
  - source: `/mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox`
  - process: `uvicorn app.main:app`
  - internal port: `5013`
  - public path option: `https://helen.mus.auth.gr/soundsketcher/`
  - subdomain option: `https://soundsketcher.mus.auth.gr/`

## Install Systemd Services

Run these on the server:

```bash
cd /mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox

sudo cp deploy/systemd/soundsketcher-refactor.service /etc/systemd/system/
sudo cp deploy/systemd/soundsketcher-legacy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable soundsketcher-refactor
sudo systemctl enable soundsketcher-legacy
sudo systemctl start soundsketcher-refactor
sudo systemctl start soundsketcher-legacy
```

If either port is already occupied by a manual `screen` or `nohup` process, stop
that manual process before starting the matching service.

Check status:

```bash
sudo systemctl status soundsketcher-refactor
sudo systemctl status soundsketcher-legacy
```

Follow logs:

```bash
sudo journalctl -u soundsketcher-refactor -f
sudo journalctl -u soundsketcher-legacy -f
```

Restart manually:

```bash
sudo systemctl restart soundsketcher-refactor
sudo systemctl restart soundsketcher-legacy
```

## Validate Internal Ports

```bash
curl -I http://127.0.0.1:5013/
curl -I http://127.0.0.1:5002/
```

## Nginx

Use the two nginx snippets in `deploy/nginx/`:

- `soundsketcher-upstreams.conf` belongs in nginx's `http` context, for example
  `/etc/nginx/conf.d/soundsketcher-upstreams.conf`.
- `soundsketcher-locations.conf` belongs inside the existing
  `helen.mus.auth.gr` `server { ... }` block.

Keep `/app1/` for legacy.  The refactor can be served under `/soundsketcher/`
when `SOUNDSKETCHER_BASE_PATH="/soundsketcher"` is present in `.env.staging`.

One practical install pattern is:

```bash
cd /mnt/ssd1/kvelenis/soundsketcher-staging/refactor_sandbox

sudo cp deploy/nginx/soundsketcher-upstreams.conf /etc/nginx/conf.d/
sudo cp deploy/nginx/soundsketcher-locations.conf /etc/nginx/snippets/
```

Then add this line inside the active `server` block for `helen.mus.auth.gr`:

```nginx
include /etc/nginx/snippets/soundsketcher-locations.conf;
```

Finally validate and reload nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
```
