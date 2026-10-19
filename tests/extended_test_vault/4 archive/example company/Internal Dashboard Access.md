## Internal dashboard access
Example Company accesses a development dashboard through an authorized bastion.
All connection names and addresses here are placeholders.

```bash
ssh -N -L 4000:<dashboard-host>:3000 <bastion-host>
```

Use the approved account and access procedure. Keep private addresses, usernames,
and credentials in local configuration rather than in a shared note.

---
Associations: [[z> MOC Infrastructure]]
