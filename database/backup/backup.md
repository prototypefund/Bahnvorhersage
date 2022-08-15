# Automatic postgres backups
The scrips in this directory are used to create automatic backups of the postgresql database. 

The Scripts are copied from https://wiki.postgresql.org/wiki/Automated_Backup_on_Linux, but `gzip` was replaced with `lz4 -3` for faster compression. Note that `lz4` must be installed:
```bash
sudo apt install lz4
```

In order to setup the scripts correctly, this artikle was used: https://medium.com/@elnur.j/setup-postgres-database-daily-backups-on-ubuntu-5c5a3c8a59a5