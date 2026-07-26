# Third-party asset clearance ledger

- Audit date: 2026-07-11
- Upstream project: [Farama Foundation Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- Pinned upstream commit: `7a1191388aa4aa973d3a5e4b039899cd99cc991f`
- Local upstream root: `ref/gymnasium`
- Hash inventory: [`THIRD_PARTY_ASSETS.sha256`](THIRD_PARTY_ASSETS.sha256)

This is an engineering clearance record, not legal advice. A file being present in the MIT-licensed
Gymnasium repository does not by itself resolve rights in artwork that Gymnasium identifies as
coming from another creator or stock-asset source.

## Decision summary

The pinned tree contains 83 non-code visual assets in the audited directories. Only one is cleared
for promotion into this repository:

- **Cleared for reuse:** Pendulum's `clockwise.png`, under Gymnasium's MIT license and notice.
- **Blocked from byte reuse:** the other 82 files. Do not copy them into a first-party asset
  directory, a crate package, generated fixtures, documentation, or release artifacts.
- **Replacement policy:** create clean, project-authored replacements for every blocked family.
  Replacements may reproduce the environment's functional composition, scale, palette role, and
  semantic silhouette, but must not trace, transform, recolor, or otherwise derive from blocked
  bytes.

No asset bytes were copied as part of this audit.

## Clearance vocabulary

- **Yes:** the evidence in this ledger grants the use, subject to the stated notice.
- **Conditional:** a published license appears to permit the use only for a lawful licensee or with
  additional written permission; this repository does not currently hold the necessary evidence.
- **No:** the published terms prohibit the use in this distribution model.
- **Not cleared:** no sufficient permission or license evidence was found. Treat this the same as
  **No** until the ledger is updated with durable evidence.

For this project, **crate inclusion** includes both Cargo source packages and compiled applications.
Assets in a Cargo package are distributed as raw files, and embedding a file in a binary does not
make an otherwise missing redistribution grant safe to assume.

## Pinned-source and integrity evidence

The audited roots are:

- `ref/gymnasium/gymnasium/envs/classic_control/assets/`
- `ref/gymnasium/gymnasium/envs/toy_text/font/`
- `ref/gymnasium/gymnasium/envs/toy_text/img/`

The adjacent SHA-256 inventory names all 83 files individually. Verify it from the repository root:

```sh
git -C ref/gymnasium rev-parse HEAD
sha256sum -c THIRD_PARTY_ASSETS.sha256
```

The first command must print the pinned commit above, and the second must report 83 `OK` results.
Any submodule update or asset-byte change invalidates this audit until the affected provenance,
license evidence, decisions, and hashes are reviewed again.

Gymnasium's pinned repository license is MIT and requires preservation of its copyright and
permission notice. The current notice names OpenAI and the Farama Foundation:

- [Pinned Gymnasium MIT license](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/LICENSE)
- Local copy: `ref/gymnasium/LICENSE`

Upstream's own source comments explicitly identify several asset families as third-party. Those
specific provenance statements take precedence over a blanket assumption that every image or font
is available under MIT.

## Rights matrix

| Family                                   | Files | Raw redistribution                                    | Crate inclusion      | Rendered-video use                               | Modification                                                      | Decision                                                   |
| ---------------------------------------- | ----: | ----------------------------------------------------- | -------------------- | ------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| Pendulum torque arrow                    |     1 | Yes, with MIT notice                                  | Yes, with MIT notice | Yes                                              | Yes                                                               | **Reuse unchanged or modified**                            |
| Blackjack card art                       |    53 | No current grant                                      | No                   | Not cleared without an independent stock license | Not cleared without an independent stock license                  | **Clean replacement**                                      |
| Blackjack `Minecraft.ttf`                |     1 | Not cleared                                           | No                   | Not cleared                                      | Not cleared                                                       | **Clean replacement**                                      |
| Franuka snow elf and stool               |     5 | No; published terms prohibit redistribution as-is     | No                   | Conditional project use for a lawful licensee    | Conditional editing; open-source redistribution remains uncleared | **Clean replacement unless written exception is obtained** |
| Franuka Taxi passenger                   |     1 | No; published terms prohibit redistribution as-is     | No                   | Conditional project use for a lawful licensee    | Conditional editing; open-source redistribution remains uncleared | **Clean replacement unless written exception is obtained** |
| Mel Tillery Frozen Lake art              |     4 | Not cleared; archived source says all rights reserved | No                   | Not cleared                                      | Not cleared                                                       | **Clean replacement**                                      |
| Mel Tillery Taxi art                     |    12 | Not cleared; archived source says all rights reserved | No                   | Not cleared                                      | Not cleared                                                       | **Clean replacement**                                      |
| Cliff Walking art with unresolved author |     6 | Not cleared                                           | No                   | Not cleared                                      | Not cleared                                                       | **Clean replacement**                                      |

## Family records

### Pendulum torque arrow — cleared

- **File:** `ref/gymnasium/gymnasium/envs/classic_control/assets/clockwise.png`
- **SHA-256:** `97b50edad865fb564c7634cc26fb5712880a16870f9f8b89b52d89f0374ac9a9`
- **Upstream use:** `gymnasium/envs/classic_control/pendulum.py`
- **History:** introduced in OpenAI Gym's initial-release commit
  `e8f29806033861f196cbc679788b985f4b40c4b7` on 2016-04-27. OpenAI added the
  repository MIT license later that day in commit
  `9c3516953d9a1d121ee13f65b3f2a11db3b04743`. No separate creator, stock source,
  or contrary asset terms appear in the pinned history or source.
- **Creator/source:** OpenAI Gym repository; individual artist not stated.
- **Known license:** Gymnasium MIT license.
- **Required attribution:** retain the pinned Gymnasium MIT notice, including `Copyright (c) 2016
  OpenAI` and `Copyright (c) 2022 Farama Foundation`, in the promoted asset's notice surface.
- **Conclusions:** raw redistribution, crate inclusion, rendered-video use, and modification are
  cleared under MIT with the required notice.
- **Decision:** **reuse unchanged** by default. Modification is also permitted if renderer needs
  require it, but preserve the original hash and source information in this ledger.

### Blackjack playing cards — blocked

- **Files:** all 52 suit/rank images matching `img/{C,D,H,S}{2,3,4,5,6,7,8,9,T,J,Q,K,A}.png`
  plus `img/Card.png` (53 PNGs total). Exact names and hashes are in the SHA-256 inventory.
- **Upstream path:** `ref/gymnasium/gymnasium/envs/toy_text/img/`
- **History:** introduced by OpenAI Gym PR
  [#2550](https://github.com/openai/gym/pull/2550), merged as commit
  `0bbef5f0fb653617c4d9599b99e9d88405ba42a5` on 2022-01-19.
- **Creator/source:** Gymnasium credits Mariia Khmelnytska and links to 123RF stock asset
  `104453049`, “Pixel art playing cards standart deck vector set,” in
  `gymnasium/envs/toy_text/blackjack.py`. The same work is listed by stock sites under contributor
  handle `kmarfu`.
- **Known license/permission:** no stock purchase receipt, license certificate, sublicense, or
  redistribution permission is bundled with Gymnasium. 123RF states that stock users license rather
  than own the content, and its current terms prohibit transfer, sublicensing, making source content
  available for download, and unauthorized redistribution. Even an end-product license is not a
  grant to publish the source images in an open Cargo package.
- **Required attribution if separately licensed:** follow the exact license certificate obtained
  for this asset and credit Mariia Khmelnytska / the applicable stock provider where required. No
  such certificate currently exists in this repository.
- **Conclusions:** raw redistribution and crate inclusion are not allowed by the evidence available.
  Rendered-video use and modification are also not cleared for this project without an independent,
  documented stock license; such a license still would not clear raw source distribution.
- **Decision:** **clean replacement**. Do not use the upstream PNG bytes even as temporary runtime
  or demo-video assets.
- **Sources:**
  [Gymnasium credit](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/toy_text/blackjack.py),
  [123RF license terms](https://www.123rf.com/license/extended/).

### Blackjack `Minecraft.ttf` — blocked

- **File:** `ref/gymnasium/gymnasium/envs/toy_text/font/Minecraft.ttf`
- **SHA-256:** `bd47314d301e50ff4d109bff28dfcf637cb7eb13945480259878b848875acc65`
- **History:** introduced by OpenAI Gym PR #2550 / commit
  `0bbef5f0fb653617c4d9599b99e9d88405ba42a5`.
- **Creator/source:** unknown. The font reports family/full name `Minecraft`, PostScript name
  `Minecraft`, and foundry `2ttf`; it contains no discoverable copyright or license metadata.
- **Known license/permission:** none. No font license or source URL is bundled in the asset
  directory, Gymnasium source, introducing PR, or pinned repository history.
- **Required attribution:** cannot be determined.
- **Conclusions:** raw redistribution, crate inclusion, rendered-video use, and modification are all
  not cleared.
- **Decision:** **clean replacement** using a project-authored bitmap font, Bevy's cleared default
  font, or a separately selected font with a durable OFL/Apache/MIT-compatible license record.

### Franuka snow elf and stool — blocked for source distribution

- **Files:** `elf_down.png`, `elf_left.png`, `elf_right.png`, `elf_up.png`, and `stool.png`.
- **Upstream path:** `ref/gymnasium/gymnasium/envs/toy_text/img/`
- **History:** introduced with Frozen Lake rendering in OpenAI Gym PR
  [#2568](https://github.com/openai/gym/pull/2568), merged as commit
  `c6b6754b128c095df49c74785277d8d5e9f81755` on 2022-02-05; reused by Cliff
  Walking.
- **Creator/source:** Franuka,
  [RPG snow tileset](https://franuka.itch.io/rpg-snow-tileset).
- **Known license/permission:** the current creator page permits commercial and non-commercial
  project use and editing, requests a creator-page/social link where possible, and prohibits
  redistributing the pack as-is or reselling it. The page invites contact for specific cases.
- **Required attribution:** the page says a link is not mandatory, but this project would credit
  `Franuka — RPG snow tileset` with the creator-page URL if a specific use were cleared.
- **Conclusions:** raw redistribution and Cargo crate inclusion are not cleared. Rendered-video use
  and editing are **conditional** for a lawful licensee, but the current repository has neither a
  purchase record nor written permission for the open-source/watch-and-video distribution model.
  Modified source files would remain risky to distribute because the public terms do not define how
  much modification escapes the redistribution restriction.
- **Decision:** **clean replacement**. Reconsider only after written permission explicitly covers
  raw open-source/Cargo redistribution, compiled applications, rendered demos, and modification.

### Franuka Taxi passenger — blocked for source distribution

- **File:** `ref/gymnasium/gymnasium/envs/toy_text/img/passenger.png`
- **SHA-256:** `58164970b0280fbfc6f428a2a58496b5affce4738bbb9bc8f47322f5b6ed2c4a`
- **History:** introduced with Taxi rendering in OpenAI Gym PR
  [#2713](https://github.com/openai/gym/pull/2713), merged as commit
  `e11231d84fd8b1386455709d77051addf8ded095` on 2022-04-02.
- **Creator/source:** Franuka,
  [RPG asset pack](https://franuka.itch.io/rpg-asset-pack).
- **Known license/permission:** the current page has the same project-use/editing grant and
  as-is-redistribution prohibition as the snow pack.
- **Required attribution:** credit `Franuka — RPG asset pack` with the creator-page URL if a
  specific use is later cleared.
- **Conclusions:** the same as the snow-assets family: no raw redistribution or crate inclusion;
  rendered-video use and modification are conditional and not currently evidenced for this project.
- **Decision:** **clean replacement**, unless written permission clears every intended distribution
  channel.

### Mel Tillery Frozen Lake art — blocked

- **Files:** `cracked_hole.png`, `goal.png`, `hole.png`, and `ice.png`.
- **Upstream path:** `ref/gymnasium/gymnasium/envs/toy_text/img/`
- **History:** introduced with Frozen Lake rendering in commit
  `c6b6754b128c095df49c74785277d8d5e9f81755`. Gym later added the creator credit in
  commit `d750eb8df0352d747b0cc0e05c323bd1b8cf25e0`, “fix some pixel art credits.”
- **Creator/source:** M. (Mel) Tillery, formerly `http://www.cyaneus.com/`, as stated in
  `gymnasium/envs/toy_text/frozen_lake.py`.
- **Known license/permission:** no reusable license or written permission is bundled. The creator's
  archived May 2022 site footer says `Copyright 2021 M. Tillery. All rights reserved.` The current
  site did not return a usable license page during this audit.
- **Required attribution if permission is later obtained:** `Art by M. Tillery`, plus whatever terms
  the creator specifies in writing.
- **Conclusions:** raw redistribution, crate inclusion, rendered-video use, and modification are all
  not cleared.
- **Decision:** **clean replacement**.
- **Archived evidence:**
  [Cyaneus homepage, 2022-05-23](https://web.archive.org/web/20220523185323/http://cyaneus.com/).

### Mel Tillery Taxi art — blocked

- **Files:** `cab_front.png`, `cab_left.png`, `cab_rear.png`,
  `cab_right.png`,
  `gridworld_median_bottom.png`, `gridworld_median_horiz.png`, `gridworld_median_left.png`,
  `gridworld_median_right.png`, `gridworld_median_top.png`, `gridworld_median_vert.png`,
  `hotel.png`, and `taxi_background.png` (12 PNGs).
- **Upstream path:** `ref/gymnasium/gymnasium/envs/toy_text/img/`
- **History:** introduced with Taxi rendering in commit
  `e11231d84fd8b1386455709d77051addf8ded095`; creator credit added in
  `d750eb8df0352d747b0cc0e05c323bd1b8cf25e0`.
- **Creator/source:** M. Tillery / Cyaneus, as stated in
  `gymnasium/envs/toy_text/taxi.py`.
- **Known license/permission and attribution:** the same unresolved, all-rights-reserved evidence as
  the Frozen Lake family.
- **Conclusions:** raw redistribution, crate inclusion, rendered-video use, and modification are all
  not cleared.
- **Decision:** **clean replacement**.
- **Archived evidence:**
  [Cyaneus games page, 2022-06-26](https://web.archive.org/web/20220626203402/http://www.cyaneus.com/games.html).

### Cliff Walking art with unresolved author — blocked

- **Files:** `cookie.png`, `mountain_bg1.png`, `mountain_bg2.png`, `mountain_cliff.png`,
  `mountain_near-cliff1.png`, and `mountain_near-cliff2.png`.
- **Upstream path:** `ref/gymnasium/gymnasium/envs/toy_text/img/`
- **History:** introduced with Cliff Walking rendering in OpenAI Gym PR
  [#2997](https://github.com/openai/gym/pull/2997), merged as commit
  `cb3df610e3a4da631714673d5069513bd4586668` on 2022-07-26. Some mountain files were
  subsequently changed in commit `43b42d52809ee3999d74d70ab13620337fc4fe45`.
- **Creator/source:** unresolved. The pinned source literally says `All other assets by ____` after
  crediting Franuka for the elf and stool. That placeholder was added in commit
  `95da6c5714c7aabbdcdec1f2e0aa57f704c77b28`, “partially fix art credits.” The
  introducing PR contains no license discussion or creator grant.
- **Known license/permission:** none sufficient.
- **Required attribution:** cannot be determined.
- **Conclusions:** raw redistribution, crate inclusion, rendered-video use, and modification are all
  not cleared.
- **Decision:** **clean replacement**.
- **Source:**
  [Pinned Cliff Walking source](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/toy_text/cliffwalking.py).

## Promotion rules

1. Never copy a blocked upstream byte into a first-party path, even temporarily.
2. Runtime and release code must never load assets from `ref/gymnasium`.
3. A clean replacement must have its own source file, creator, creation date, SHA-256, and explicit
   license recorded here before use.
4. Project-authored replacements should be committed under this repository's dual license only when
   the author has explicitly agreed to that license.
5. Third-party replacements should prefer CC0, MIT, Apache-2.0, or OFL-1.1 as appropriate. Record
   attribution, notice, modification, font-embedding, video, and source-redistribution requirements
   separately.
6. Do not infer permission from an upstream screenshot, package install, Git history, a public URL,
   or lack of a copyright notice.
7. Written exceptions must identify the exact files and permit raw public-source redistribution,
   Cargo/crate and compiled-app inclusion, rendered videos/screenshots, and modification. Preserve
   the original message or signed grant outside the repository and record a stable evidence pointer
   here.
8. Re-run the hash check and manually verify this ledger before any renderer PR promotes assets.

## Audit sources

- [Pinned Gymnasium MIT license](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/LICENSE)
- [Pinned Blackjack source and stock-art credit](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/toy_text/blackjack.py)
- [Pinned Frozen Lake source and credits](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/toy_text/frozen_lake.py)
- [Pinned Cliff Walking source and unresolved credit](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/toy_text/cliffwalking.py)
- [Pinned Taxi source and credits](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/toy_text/taxi.py)
- [Franuka RPG snow tileset terms](https://franuka.itch.io/rpg-snow-tileset)
- [Franuka RPG asset pack terms](https://franuka.itch.io/rpg-asset-pack)
- [123RF licensing terms](https://www.123rf.com/license/extended/)
- [Archived Cyaneus homepage](https://web.archive.org/web/20220523185323/http://cyaneus.com/)
- [Archived Cyaneus games page](https://web.archive.org/web/20220626203402/http://www.cyaneus.com/games.html)
