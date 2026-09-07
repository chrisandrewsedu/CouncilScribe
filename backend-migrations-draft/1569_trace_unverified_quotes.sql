-- 1569_trace_unverified_quotes.sql
--
-- DRAFT — NOT APPLIED. Scratch path in the on-the-record repo; a human moves this to
-- ev-accounts after review. Generated 2026-08-07.
--
-- Re-sources the 10 TRACED quotes among the 16 that the 2026-08-07 comparability audits
-- flagged source-unverified / source-speaker-mismatch and that the meetings.segments sweep
-- could not resolve. The other 6 are UNRESOLVED and are handled, commented out, in
-- SECTION 2 below — a human decides retirement or deletion, not this migration.
--
-- Full evidence, per quote, including the verbatim run matched:
--   docs/audits/2026-08-07-unverified-quote-trace.md
--
-- Confirmation standard: a distinctive contiguous run of each quote's own words appears in
-- the confirmed source. Exact contiguous runs by row: 24, 15, 13, 11, 10, 10, 9, and the
-- full sentence for the two press-statement rows. One row (Becerra / tariffs) has a strictly
-- exact run of only 6 words and is included on the strength of a 13-word near-identical span
-- plus moderator-named speaker at a fixed timestamp — see CAVEAT on that statement.
--
-- DELIBERATELY NOT SET: editor_note. House style plus human sign-off required; see the
-- findings doc for the elisions and text corrections each note has to justify.
--
-- >>> READ BEFORE APPLYING <<<
-- Three rows updated here still must NOT ship as written, because the quote TEXT is defective
-- even though the source is now correct. This migration does not touch quote_text; correcting
-- it is a separate, human editorial act.
--   * ebb39e53 (Hilton/abortion) — stored "I don't want..."; audio says "We don't want...".
--   * e842e9d2 (Becerra/tariffs) — stored "tariffs that are attacks"; audio is "tariffs that
--     are a tax". The stored text reproduces an ASR homophone error and is not grammatical.
--   * 97459d18 / ea35df13 (Raman) — text truncates mid-sentence at "...America." where the
--     original continues ", and our affordability crisis is making it even worse".


BEGIN;

-- ---------------------------------------------------------------------------
-- Guard: the 10 target rows must still carry the source_url this trace started
-- from. If curation moved underneath this migration, abort rather than overwrite.
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    expected CONSTANT integer := 10;
    found integer;
BEGIN
    SELECT count(*) INTO found
      FROM essentials.quotes
     WHERE (id, source_url) IN (
               ('41ac4890-8de7-46b3-8e85-b62a99bdf5f5'::uuid, 'https://lapublicpress.org/2026/04/la-budget-2026-bass/'),
               ('17d7f049-984a-4d36-b4dd-dab0dcab0069'::uuid, 'https://mayor.lacity.gov/news/delivering-results-2024-bass-highlights-unprecedented-green-year-la'),
               ('ef8712c2-f45c-4905-a177-bab2285e6a89'::uuid, 'https://cityclerk.lacity.org/lacityclerkconnect/index.cfm?fa=ccfi.viewrecord&cfnumber=25-0002-S8'),
               ('8403a778-c94c-4bd1-971d-76ae1abdaf8c'::uuid, 'https://cityclerk.lacity.org/lacityclerkconnect/index.cfm?fa=ccfi.viewrecord&cfnumber=21-0002-S55'),
               ('97459d18-b3dd-44d4-8648-8358c18969dd'::uuid, 'https://cityclerk.lacity.org/lacityclerkconnect/index.cfm?fa=ccfi.viewrecord&cfnumber=21-0972'),
               ('ea35df13-e121-41e5-9a0d-fc63ffdfd06f'::uuid, 'https://cityclerk.lacity.org/lacityclerkconnect/index.cfm?fa=ccfi.viewrecord&cfnumber=21-0972'),
               ('9eb66701-bd1f-4479-8d6b-c4ec1a484ffa'::uuid, 'https://www.youtube.com/watch?v=UUOsiG5tkDU'),
               ('ebb39e53-3e7f-4ecd-b982-a45db47242e7'::uuid, 'https://www.youtube.com/watch?v=qRNZ0kuA49k'),
               ('9403fcba-bb26-425f-bf27-997a4667b05d'::uuid, 'https://www.youtube.com/watch?v=qRNZ0kuA49k'),
               ('e842e9d2-abcb-4fb1-8131-c8f4bd4cbbe5'::uuid, 'https://www.youtube.com/watch?v=qRNZ0kuA49k')
           );

    IF found <> expected THEN
        RAISE EXCEPTION
            'Aborting: expected % TRACED rows still carrying their pre-trace source_url, found %.'
            ' Re-run the trace before applying.', expected, found;
    END IF;
END $$;


-- ===========================================================================
-- SECTION 1 — TRACED. Re-source to the confirmed origin.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- LIVE (readrank_selected = true) — visible to citizens now. Apply first.
-- ---------------------------------------------------------------------------

-- Karen Bass | public-safety-approach | LIVE
-- LA Magazine, 2025-12-11, "Karen Bass Warns of LAPD Staffing Low" (Michele McPhee),
-- quoting Bass's letter to the City Council.
-- Verbatim run: "The second largest city in the United States cannot have an effective
--               police department with 8,300 officers…"  (in quotation marks, on the page)
-- Was cited to an April 2026 LA Public Press budget article — wrong story, two months adrift.
UPDATE essentials.quotes SET
    source_url  = 'https://lamag.com/news/mayor-karen-bass-pushes-city-council-to-approve-hiring-of-more-lapd-cops/',
    source_name = 'lamag.com'
  WHERE id = '41ac4890-8de7-46b3-8e85-b62a99bdf5f5';

-- ---------------------------------------------------------------------------
-- Draft rows (readrank_selected = false)
-- ---------------------------------------------------------------------------

-- Karen Bass | housing
-- Office of the Mayor press release, April 2026 (Adaptive Reuse Ordinance / DTLA WTC).
-- Verbatim run: the full quote, exact, attributed "said Mayor Bass".
-- Was cited to the "Delivering Results 2024" release, which contains neither
-- "status quo" nor "stunted".
UPDATE essentials.quotes SET
    source_url  = 'https://mayor.lacity.gov/news/mayor-bass-visits-adaptive-reuse-project-will-create-more-500-units-affordable-housing',
    source_name = 'mayor.lacity.gov'
  WHERE id = '17d7f049-984a-4d36-b4dd-dab0dcab0069';

-- Nithya Raman | homelessness
-- CD4 press release, remarks on revised LAMC 41.18, via Wayback capture 2022-08-19.
-- The retired councildistrict4.lacity.org host now 301-redirects; the archived capture is
-- the citable artefact. Verbatim run (24 words, exact):
--   "This creates a district by district arms-race, where people will get pushed around from
--    district to district instead of having a citywide strategy that prioritizes intervention
--    in encampments by need, by safety, by fire risk"
-- Was cited to council file 25-0002-S8 (a legislative index page; carries no quotations).
UPDATE essentials.quotes SET
    source_url  = 'http://web.archive.org/web/20220819070334/https://councildistrict4.lacity.org/councilmember-nithya-raman-remarks-todays-la-city-council-meeting-revised-city-ordinance-4118',
    source_name = 'web.archive.org (councildistrict4.lacity.org, captured 2022-08-19)'
  WHERE id = 'ef8712c2-f45c-4905-a177-bab2285e6a89';

-- Nithya Raman | deportation
-- Campaign platform page "Immigrants In, ICE Out".
-- Verbatim run: the entire quote, exact (9 words).
-- Was cited to council file 21-0002-S55 (US Citizenship Act 2021 — moved by Nury Martinez;
-- Raman is not even a mover on that file).
-- NOTE FOR REVIEW: this is published platform prose, not speech. Confirm in scope.
UPDATE essentials.quotes SET
    source_url  = 'https://www.nithyaforthecity.com/immigrants',
    source_name = 'www.nithyaforthecity.com'
  WHERE id = '8403a778-c94c-4bd1-971d-76ae1abdaf8c';

-- Nithya Raman | civil-rights
-- Streetsblog LA, 2021-10-28, reporting Raman's press statement on the Affordable Housing
-- Overlay Zone motion. Verbatim run (10 words, exact):
--   "L.A. is one of the most segregated cities in America"
-- Was cited to council file 21-0972 (the correct underlying motion, but an index page).
-- CAVEAT: stored text truncates at "America." where the original continues
--   ", and our affordability crisis is making it even worse". editor_note must justify this.
UPDATE essentials.quotes SET
    source_url  = 'https://la.streetsblog.org/2021/10/28/council-approves-raman-harris-dawson-motion-to-foster-affordable-development-in-high-resource-areas',
    source_name = 'la.streetsblog.org'
  WHERE id = '97459d18-b3dd-44d4-8648-8358c18969dd';

-- Nithya Raman | housing  (identical quote text to 97459d18; same source, same caveat)
UPDATE essentials.quotes SET
    source_url  = 'https://la.streetsblog.org/2021/10/28/council-approves-raman-harris-dawson-motion-to-foster-affordable-development-in-high-resource-areas',
    source_name = 'la.streetsblog.org'
  WHERE id = 'ea35df13-e121-41e5-9a0d-fc63ffdfd06f';

-- Steve Hilton | fossil-fuels
-- The CITED VIDEO WAS ALREADY CORRECT (NBCLA, "Full NBC4 broadcast: 2026 California governor
-- candidates"). Only the timestamp was missing; YouTube's own caption track contains it at
-- 00:20:26, which the ingested transcript did not. Verbatim run (13 words, exact):
--   "I'll get it done directly as governor by instructing the California Department of"
UPDATE essentials.quotes SET
    source_url  = 'https://www.youtube.com/watch?v=UUOsiG5tkDU&t=1226',
    source_name = 'www.youtube.com'
  WHERE id = '9eb66701-bd1f-4479-8d6b-c4ec1a484ffa';

-- ---------------------------------------------------------------------------
-- The three rows below were all cited to https://www.youtube.com/watch?v=qRNZ0kuA49k
-- (KTLA 5, "California Governor's Debate - Full Broadcast"). That video has NO caption
-- track of any kind (yt-dlp: "has no automatic captions / has no subtitles"), which is why
-- neither the audit nor the transcript sweep could verify anything against it.
-- The actual event is the CBS News California / San Francisco Examiner gubernatorial
-- debate of 2026-05-15. Timestamps below are into the Face the Nation full-video upload
-- (xFNkHY_m_eE). The same debate is also at E77UzwNExAA (CBS News Sacramento, longer cut
-- including ~22.6 minutes of pre-show, so add ~1357s to these offsets) and was used as an
-- independent second caption track for corroboration.
-- ---------------------------------------------------------------------------

-- Steve Hilton | abortion  @ 01:14:11
-- Verbatim run (11 words, exact):
--   "don't want Louisiana dictating our laws. We shouldn't be dictating Louisiana's"
-- Speaker confirmed: moderator addresses Hilton immediately before and after the turn.
-- >>> TEXT DEFECT: stored text opens "I don't want"; audio says "We don't want". FIX BEFORE SHIPPING.
UPDATE essentials.quotes SET
    source_url  = 'https://www.youtube.com/watch?v=xFNkHY_m_eE&t=4451',
    source_name = 'www.youtube.com'
  WHERE id = 'ebb39e53-3e7f-4ecd-b982-a45db47242e7';

-- Xavier Becerra | abortion  @ 01:13:29
-- Verbatim run (15 words — the entire quote, exact):
--   "absolutely no and when i was ag i protected with reproductive rights here in california"
-- Speaker confirmed: moderator calls "Mr. Becerra" immediately before; "when I was AG" is
-- self-identifying (CA Attorney General 2017-2021). Corroborated at E77UzwNExAA @ 01:36:05.
UPDATE essentials.quotes SET
    source_url  = 'https://www.youtube.com/watch?v=xFNkHY_m_eE&t=4409',
    source_name = 'www.youtube.com'
  WHERE id = '9403fcba-bb26-425f-bf27-997a4667b05d';

-- Xavier Becerra | tariffs  @ 00:21:04
-- CAVEAT — WEAKEST ROW IN THIS MIGRATION. Strictly exact contiguous run is only 6 words
-- ("we're going to fight against Trump"), which is not distinctive on its own. Included
-- because the enclosing 13-word span matches near-identically in BOTH caption tracks and the
-- moderator names Becerra immediately after the turn. If a reviewer wants to hold this row
-- to the exact-run bar alone, retire it instead — that is a defensible call.
-- >>> TEXT DEFECT: stored "the tariffs that are attacks"; audio is "tariffs that are a tax".
--     The stored text reproduces an ASR homophone error and is not grammatical English.
--     Re-transcribe from the timestamp or retire. FIX BEFORE SHIPPING.
UPDATE essentials.quotes SET
    source_url  = 'https://www.youtube.com/watch?v=xFNkHY_m_eE&t=1264',
    source_name = 'www.youtube.com'
  WHERE id = 'e842e9d2-abcb-4fb1-8131-c8f4bd4cbbe5';


-- ---------------------------------------------------------------------------
-- Guard: exactly 10 rows must now carry their new source_url.
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    expected CONSTANT integer := 10;
    found integer;
BEGIN
    SELECT count(*) INTO found
      FROM essentials.quotes
     WHERE (id, source_url) IN (
               ('41ac4890-8de7-46b3-8e85-b62a99bdf5f5'::uuid, 'https://lamag.com/news/mayor-karen-bass-pushes-city-council-to-approve-hiring-of-more-lapd-cops/'),
               ('17d7f049-984a-4d36-b4dd-dab0dcab0069'::uuid, 'https://mayor.lacity.gov/news/mayor-bass-visits-adaptive-reuse-project-will-create-more-500-units-affordable-housing'),
               ('ef8712c2-f45c-4905-a177-bab2285e6a89'::uuid, 'http://web.archive.org/web/20220819070334/https://councildistrict4.lacity.org/councilmember-nithya-raman-remarks-todays-la-city-council-meeting-revised-city-ordinance-4118'),
               ('8403a778-c94c-4bd1-971d-76ae1abdaf8c'::uuid, 'https://www.nithyaforthecity.com/immigrants'),
               ('97459d18-b3dd-44d4-8648-8358c18969dd'::uuid, 'https://la.streetsblog.org/2021/10/28/council-approves-raman-harris-dawson-motion-to-foster-affordable-development-in-high-resource-areas'),
               ('ea35df13-e121-41e5-9a0d-fc63ffdfd06f'::uuid, 'https://la.streetsblog.org/2021/10/28/council-approves-raman-harris-dawson-motion-to-foster-affordable-development-in-high-resource-areas'),
               ('9eb66701-bd1f-4479-8d6b-c4ec1a484ffa'::uuid, 'https://www.youtube.com/watch?v=UUOsiG5tkDU&t=1226'),
               ('ebb39e53-3e7f-4ecd-b982-a45db47242e7'::uuid, 'https://www.youtube.com/watch?v=xFNkHY_m_eE&t=4451'),
               ('9403fcba-bb26-425f-bf27-997a4667b05d'::uuid, 'https://www.youtube.com/watch?v=xFNkHY_m_eE&t=4409'),
               ('e842e9d2-abcb-4fb1-8131-c8f4bd4cbbe5'::uuid, 'https://www.youtube.com/watch?v=xFNkHY_m_eE&t=1264')
           );

    IF found <> expected THEN
        RAISE EXCEPTION 'Aborting: expected % re-sourced rows, found %.', expected, found;
    END IF;
END $$;

-- Guard: this migration must not have touched readrank_selected. Exactly one of the ten
-- (the Bass public-safety row) was live going in, and it must still be live going out.
DO $$
DECLARE
    live_count integer;
BEGIN
    SELECT count(*) INTO live_count
      FROM essentials.quotes
     WHERE readrank_selected
       AND id IN (
               '41ac4890-8de7-46b3-8e85-b62a99bdf5f5',
               '17d7f049-984a-4d36-b4dd-dab0dcab0069',
               'ef8712c2-f45c-4905-a177-bab2285e6a89',
               '8403a778-c94c-4bd1-971d-76ae1abdaf8c',
               '97459d18-b3dd-44d4-8648-8358c18969dd',
               'ea35df13-e121-41e5-9a0d-fc63ffdfd06f',
               '9eb66701-bd1f-4479-8d6b-c4ec1a484ffa',
               'ebb39e53-3e7f-4ecd-b982-a45db47242e7',
               '9403fcba-bb26-425f-bf27-997a4667b05d',
               'e842e9d2-abcb-4fb1-8131-c8f4bd4cbbe5'
           );

    IF live_count <> 1 THEN
        RAISE EXCEPTION
            'Aborting: expected exactly 1 live row among the 10 TRACED quotes, found %.', live_count;
    END IF;
END $$;

COMMIT;


-- ===========================================================================
-- SECTION 2 — UNRESOLVED. NOT part of the transaction above. NOTHING BELOW RUNS.
--
-- Six quotes whose text could not be confirmed in any source. Per the trace brief,
-- UNRESOLVED beats a plausible guess: none of these gets a source it did not earn.
--
-- RECOMMENDATION: RETIRE ALL SIX.
-- A human decides deletion. The statements below are commented out deliberately — they
-- de-select the two LIVE rows so they stop being shown to citizens, which is the minimum
-- safe action and is reversible. Hard deletion is a separate, human decision.
--
-- Evidence and the full list of searches run per quote:
--   docs/audits/2026-08-07-unverified-quote-trace.md
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- PRIORITY 1 — LIVE AND FABRICATED. In front of voters right now.
-- These two are not mis-citations. The words were never said by the named candidate.
-- ---------------------------------------------------------------------------

-- Nithya Raman | growth-and-development | *** LIVE ***
--   "[Measure ULA has] become a major obstacle [to new housing development]."
--   cited: https://www.nithyaforthecity.com/housing  (contains no instance of "obstacle",
--   live or in any of its 7 Wayback captures; nor does the CD4 ULA-reform release)
--
-- NOT A QUOTATION. Traced to its origin: the LA Times (2026-02-15) wrote, with NO quotation
-- marks anywhere, "Although she has been a supporter of the tax, she has also concluded that
-- it is a major obstacle to building new housing" — the reporter's own paraphrase.
-- Wikipedia then re-wrote that sentence and ADDED quotation marks around "major obstacle";
-- American Kahani copied Wikipedia; our curator reconstructed a first-person sentence from
-- the reported speech using square brackets. Raman's own words survive nowhere in the chain.
--
-- This is the same aggregator failure mode as the 2026-07-25 ontheissues/wikipedia purge.
-- Recommend RETIRE (then delete).
--
-- UPDATE essentials.quotes SET readrank_selected = false
--   WHERE id = '8f51a4e3-954b-48b9-acb0-71a437bf8b08';

-- Karen Bass | homelessness | *** LIVE ***
--   "I will maintain focus on interim housing until the last street encampment is gone."
--   cited: https://mayor.lacity.gov/InsideSafe
--
-- No source found. The cited page is a static programme description carrying no first-person
-- Bass quotation, live or in any archived capture (3 snapshots checked). Five distinct
-- phrase searches returned nothing containing any contiguous run of this text. The nearest
-- real Bass statements are differently worded and cannot substitute.
-- Recommend RETIRE (then delete unless a curator can source it).
--
-- UPDATE essentials.quotes SET readrank_selected = false
--   WHERE id = '1a1a9e98-f428-43e0-a35d-f341e0f07510';

-- ---------------------------------------------------------------------------
-- PRIORITY 2 — draft rows (readrank_selected already false; no de-selection needed).
-- Recommend RETIRE — do not promote these to live.
-- ---------------------------------------------------------------------------

-- Xavier Becerra | abortion
--   "We protect the women of the state 67% voted for a woman's right to choose."
--   cited: https://www.youtube.com/watch?v=qRNZ0kuA49k
--
-- WRONG SPEAKER. The material IS in the 2026-05-15 CBS/SF Examiner debate at 01:13:19, but
-- the speaker is Antonio Villaraigosa, not Becerra. Two independent caption tracks agree on
-- the turn structure: the moderator thanks Villaraigosa, THEN calls Becerra, who answers
-- separately at 01:13:29 (that answer is quote 9403fcba, re-sourced in SECTION 1). Classic
-- off-by-one capture of the line preceding the target candidate's turn.
-- Villaraigosa is not a candidate in either race in scope, so re-attribution does not save it.
-- The stored text is also not a faithful rendering (longest exact run: 7 words).
-- Recommend RETIRE (then delete).

-- Xavier Becerra | deportation
--   "Support DACA, oppose Muslim ban and family separation"
--   cited: https://www.xavierbecerra2026.com/immigration
--
-- NOT A QUOTATION, AND THE CITED PAGE NEVER EXISTED. The URL returns HTTP 404 live (domain
-- root returns 200), and the Wayback CDX index for the whole xavierbecerra2026.com domain
-- holds captures of the ROOT PAGE ONLY across 2025-2026 — no /immigration subpage has ever
-- been archived. The text is an ontheissues.org section HEADING (verified in two archived
-- captures), written by that site's editors to label a position; the body beneath it
-- summarises CA AG press releases. Becerra never said these words in any register.
-- Survived the 2026-07-25 aggregator purge only because it carries a plausible-looking
-- campaign URL instead of the aggregator URL it actually came from.
-- Recommend RETIRE (then delete).

-- Karen Bass | public-safety-approach
--   "We are days away from the World Cup and other international events. L.A., we know, is
--    under-policed and that's not an option."
--   cited: https://nbclosangeles.com/news/local/mayor-bass-lapd-funding-recruitment/3814358/
--
-- No source found. The cited page fetches fine and does cover Bass/LAPD/World Cup, but
-- contains none of "under-policed", "not an option" or "days away" — it is the December 2025
-- $4.4M-letter story, while "days away from the World Cup" implies ~June 2026. Five phrase
-- searches plus seven news pages probed, all negative. One page could not be checked:
-- lapublicpress.org returns HTTP 403 to automated fetching and has no Wayback capture — a
-- human with a browser should look there before deleting.
-- Recommend RETIRE pending that one check.

-- Nithya Raman | immigration
--   "At a time when the rights of immigrants in this country are under daily attack, Los
--    Angeles has the opportunity to define what a better future for our nation could look like."
--   cited: https://cityclerk.lacity.org/...&cfnumber=21-0002-S55
--
-- No source found. The cited council file is the US Citizenship Act of 2021 resolution, moved
-- by Nury Martinez and seconded by Gilbert Cedillo — Raman is not a mover, and council-file
-- pages carry no quotations in any case. Four phrase searches negative; the campaign
-- immigrants page (live + 3 captures), two CD4 sanctuary-city pages, the CD4 immigrant-rights
-- page, and ALL 193 distinct archived pages of the retired councildistrict4.lacity.org site
-- were full-text searched — zero hits. (That same sweep is what found the source for
-- ef8712c2, so the method works.) The phrase "under daily attack" also does not fit
-- February 2021, when that resolution was introduced.
-- Recommend RETIRE (then delete).


-- ===========================================================================
-- FOLLOW-UP, not actioned here
--
-- The KTLA upload https://www.youtube.com/watch?v=qRNZ0kuA49k has no caption track of any
-- kind and was never a viable source for any quote. Four rows cited it; three are re-sourced
-- above and one turned out to be a different speaker entirely. Whatever import step attached
-- qRNZ0kuA49k to these quotes should be treated as suspect for ANY other quote citing it:
--
--   SELECT id, politician_id, topic_key, quote_text, readrank_selected
--     FROM essentials.quotes
--    WHERE source_url LIKE '%qRNZ0kuA49k%';
--
-- Separately: e842e9d2 stores an ASR homophone error verbatim ("attacks" for "a tax"), which
-- suggests at least some quotes were lifted from machine transcripts without being heard.
-- Worth a targeted sweep of debate-sourced rows for similar artefacts.
-- ===========================================================================
