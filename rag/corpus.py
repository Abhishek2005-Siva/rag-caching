"""A tiny knowledge base for a fictional cloud-notes app called "Nimbus".

Small enough to read at a glance, varied enough that dense (meaning-based) and
sparse (keyword-based) retrieval disagree in interesting ways.
"""

DOCUMENTS = [
    {
        "id": "billing-01",
        "text": "To change your Nimbus subscription plan, open Settings, then "
                "Billing, and choose Upgrade or Downgrade. Changes take effect "
                "at the start of the next billing cycle.",
    },
    {
        "id": "auth-01",
        "text": "If you forgot your password, click 'Reset password' on the "
                "sign-in screen. We email a secure link that lets you set a new "
                "password. The link expires after 30 minutes.",
    },
    {
        "id": "auth-02",
        "text": "Two-factor authentication adds a second step when you log in. "
                "Enable it under Settings, Security. You can use an authenticator "
                "app or SMS codes.",
    },
    {
        "id": "sync-01",
        "text": "Nimbus syncs your notes across devices automatically. If a note "
                "is missing, check that you are signed in to the same account and "
                "that sync is enabled in Settings.",
    },
    {
        "id": "export-01",
        "text": "You can export notes as Markdown or PDF. Open a note, click the "
                "three-dot menu, and choose Export. Bulk export of a whole "
                "notebook is available on paid plans.",
    },
    {
        "id": "share-01",
        "text": "Share a note by clicking Share and entering an email address. "
                "You can grant view-only or edit access. Shared collaborators "
                "need a free Nimbus account to open the note.",
    },
    {
        "id": "storage-01",
        "text": "Free accounts include 2 GB of storage. Attachments and images "
                "count toward this limit. Upgrade to Pro for 100 GB if you run "
                "out of space.",
    },
    {
        "id": "offline-01",
        "text": "Nimbus works offline. Notes you edit without a connection are "
                "saved locally and uploaded the next time you reconnect. Look for "
                "the cloud icon to confirm a note has synced.",
    },
]

# Convenience lookup for pretty printing.
BY_ID = {d["id"]: d for d in DOCUMENTS}
