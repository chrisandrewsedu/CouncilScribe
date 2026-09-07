-- 1566_backfill_la_mayor_debate_provenance.sql
--
-- DRAFT — NOT APPLIED. Scratch path in the on-the-record repo; a human moves this to
-- ev-accounts after review. Generated 2026-08-07.
--
-- Backfills source_url / source_name for the 30 orphaned Karen Bass and Nithya Raman
-- quotes (source_url IS NULL, blank source_name, blank editor_note, created_at IS NULL).
--
-- All 30 were confirmed, quote by quote, against two independent transcripts of one event:
--   NBC4 Los Angeles / Telemundo 52 mayoral primary debate, 2026-05-06
--   (Karen Bass, Nithya Raman, Spencer Pratt)
-- already ingested as meetings.meetings f2cf80ef-a811-4d95-990d-b9c598284eb6
-- from https://www.youtube.com/watch?v=8rI3A6alVHM
--
-- Confirmation standard: a distinctive contiguous run of each quote's own words appears in
-- the source (shortest 12 words, median 45.5, longest 100), present in BOTH transcripts, with
-- meetings.segments speaker diarization naming the expected candidate in all 30 cases.
-- Evidence per quote, including the verbatim run matched:
--   docs/audits/2026-08-07-la-mayor-orphan-quote-provenance.md
--
-- URL form matches the four Bass/Raman quotes that already cite this meeting
-- (e.g. ...?t=2738#seg-182): t = floor(segment start_time), anchor = segment_index.
--
-- DELIBERATELY NOT SET: editor_note. House style plus human sign-off required; see the
-- findings doc for the elisions each note has to justify.
--
-- Read the Concerns section of the findings doc before applying: three of these rows
-- duplicate already-sourced quotes citing the same debate, and two pairs overlap each other.

BEGIN;

-- Guard: the 30 target rows must still be orphaned and must be the candidates we expect.
-- If curation moved underneath this migration, abort rather than overwrite.
DO $$
DECLARE
    expected CONSTANT integer := 30;
    found integer;
BEGIN
    SELECT count(*) INTO found
      FROM essentials.quotes
     WHERE id IN (
               '2638f09b-b695-4c5b-a29c-a3c7cb739a10',
               'ce9bb5b9-ac51-4eec-8765-2ebe783e316d',
               'e294ef7e-5f85-42d6-9b89-cf018e817001',
               '8418a331-2eeb-418d-8473-fd36d5671276',
               'dd5bdce9-667c-4891-955a-98ec6c49d44e',
               '4c389a82-e480-4e6e-b4f4-fe8576ced385',
               'e0ec0298-540a-4ca7-be15-d8c07d4f1412',
               'e606a5f8-ff2f-4150-8d1b-60b81a4e8652',
               '0a1aacc9-3044-4b79-9e7a-a6095cc05824',
               '78ba1ed3-e7b3-471c-911b-4e4d49d87cde',
               '368cfab9-97e5-4ad6-afbf-477b4633e1cc',
               '132093b7-7acd-437f-a2fd-cee5af8f2704',
               '8edf4cb0-ed77-4e93-8ba0-a8dbdc04a09e',
               '71044f3e-380e-4887-9169-95fbd7926f08',
               '01c51a86-040c-414a-bdc4-c978d8b76c8e',
               'a6bb4672-eaf0-4f93-991c-040440c1394c',
               'f7625f7b-634f-49d4-9155-ab38455ba400',
               '644bc8e7-4545-4f50-b40e-0c2b6ae0a055',
               '9ea38e55-6e78-4ec1-b41c-eb74fa6a5217',
               '7d2e49cd-81bc-411b-bef6-498552588a1e',
               'ee3427dd-f91a-459f-8712-98117e3e428c',
               'c31d035f-deb2-4472-8b18-51e42d841244',
               'b9139e1d-824f-4670-b46c-caa624b2a83d',
               'baa16620-68a5-465e-85b9-71c9b4f5b6f0',
               'bdc80d45-40c2-4f1d-a0ef-e5619aab2d74',
               'eaaa2cdb-c56d-4be3-a798-b4e4e85c4836',
               '6c92a9fc-0082-4612-8f22-7d5f97097c09',
               'd7490082-efcf-4820-b571-8dc9a642f782',
               'd8f9bf5d-0a2b-45e4-b98f-d186f401c73c',
               '84f56b9d-ca40-4268-b02a-81d4ac116ce0'
           )
       AND source_url IS NULL
       AND politician_id IN ('21c9e711-fb18-4afb-884f-08acd2b598ba', '26dbe16a-9dff-42c0-939f-5b5e529063ca');

    IF found <> expected THEN
        RAISE EXCEPTION
            'Aborting: expected % orphaned Bass/Raman quotes with source_url IS NULL, found %.'
            ' Re-run the audit before applying.', expected, found;
    END IF;
END $$;


-- ---------------------------------------------------------------------------
-- Event: NBC4 Los Angeles / Telemundo 52 LA mayoral debate, 2026-05-06
-- ---------------------------------------------------------------------------

-- ===== Karen Ruth Bass =====

-- city-sanitation | seg-453 @ 5293s | 32-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=5293#seg-453',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '2638f09b-b695-4c5b-a29c-a3c7cb739a10';

-- economic-development | seg-395 @ 4882s | 54-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4882#seg-395',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'ce9bb5b9-ac51-4eec-8765-2ebe783e316d';

-- growth-and-development | seg-391 @ 4806s | 53-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4806#seg-391',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'e294ef7e-5f85-42d6-9b89-cf018e817001';

-- homelessness | seg-126 @ 2074s | 38-word verbatim run | similarity 0.961
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2074#seg-126',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '8418a331-2eeb-418d-8473-fd36d5671276';

-- homelessness | seg-207 @ 3066s | 19-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3066#seg-207',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'dd5bdce9-667c-4891-955a-98ec6c49d44e';

-- homelessness-response | seg-238 @ 3159s | 41-word verbatim run | similarity 0.994
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3159#seg-238',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '4c389a82-e480-4e6e-b4f4-fe8576ced385';

-- homelessness-response | seg-273 @ 3551s | 88-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3551#seg-273',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'e0ec0298-540a-4ca7-be15-d8c07d4f1412';

-- homelessness-response | seg-301 @ 3880s | 65-word verbatim run | similarity 0.995
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3880#seg-301',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'e606a5f8-ff2f-4150-8d1b-60b81a4e8652';

-- housing | seg-305 @ 3940s | 76-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3940#seg-305',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '0a1aacc9-3044-4b79-9e7a-a6095cc05824';

-- housing | seg-321 @ 4082s | 79-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4082#seg-321',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '78ba1ed3-e7b3-471c-911b-4e4d49d87cde';

-- public-safety-approach | seg-162 @ 2540s | 35-word verbatim run | similarity 0.741
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2540#seg-162',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '368cfab9-97e5-4ad6-afbf-477b4633e1cc';

-- public-safety-approach | seg-197 @ 2989s | 57-word verbatim run | similarity 0.995
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2989#seg-197',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '132093b7-7acd-437f-a2fd-cee5af8f2704';

-- rent-regulation | seg-305 @ 3940s | 26-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3940#seg-305',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '8edf4cb0-ed77-4e93-8ba0-a8dbdc04a09e';

-- residential-zoning | seg-352 @ 4277s | 31-word verbatim run | similarity 0.741
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4277#seg-352',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '71044f3e-380e-4887-9169-95fbd7926f08';

-- ===== Nithya Raman =====

-- campaign-finance | seg-186 @ 2861s | 55-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2861#seg-186',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '01c51a86-040c-414a-bdc4-c978d8b76c8e';

-- city-sanitation | seg-358 @ 4521s | 12-word verbatim run | similarity 0.864
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4521#seg-358',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'a6bb4672-eaf0-4f93-991c-040440c1394c';

-- economic-development | seg-358 @ 4521s | 89-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4521#seg-358',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'f7625f7b-634f-49d4-9155-ab38455ba400';

-- economic-development | seg-400 @ 4996s | 75-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4996#seg-400',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '644bc8e7-4545-4f50-b40e-0c2b6ae0a055';

-- growth-and-development | seg-354 @ 4338s | 44-word verbatim run | similarity 0.997
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4338#seg-354',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '9ea38e55-6e78-4ec1-b41c-eb74fa6a5217';

-- homelessness | seg-209 @ 3072s | 12-word verbatim run | similarity 0.966
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3072#seg-209',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '7d2e49cd-81bc-411b-bef6-498552588a1e';

-- homelessness | seg-213 @ 3092s | 20-word verbatim run | similarity 0.923
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3092#seg-213',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'ee3427dd-f91a-459f-8712-98117e3e428c';

-- homelessness-response | seg-241 @ 3240s | 34-word verbatim run | similarity 0.993
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3240#seg-241',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'c31d035f-deb2-4472-8b18-51e42d841244';

-- homelessness-response | seg-276 @ 3636s | 73-word verbatim run | similarity 0.997
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=3636#seg-276',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'b9139e1d-824f-4670-b46c-caa624b2a83d';

-- housing | seg-307 @ 4006s | 34-word verbatim run | similarity 0.777
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4006#seg-307',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'baa16620-68a5-465e-85b9-71c9b4f5b6f0';

-- housing | seg-307 @ 4006s | 73-word verbatim run | similarity 0.999
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4006#seg-307',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'bdc80d45-40c2-4f1d-a0ef-e5619aab2d74';

-- local-environment | seg-157 @ 2412s | 100-word verbatim run | similarity 1.000
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2412#seg-157',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'eaaa2cdb-c56d-4be3-a798-b4e4e85c4836';

-- public-safety-approach | seg-186 @ 2861s | 38-word verbatim run | similarity 0.994
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2861#seg-186',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '6c92a9fc-0082-4612-8f22-7d5f97097c09';

-- public-safety-approach | seg-193 @ 2918s | 47-word verbatim run | similarity 0.989
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=2918#seg-193',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'd7490082-efcf-4820-b571-8dc9a642f782';

-- residential-zoning | seg-354 @ 4338s | 56-word verbatim run | similarity 0.940
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4338#seg-354',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = 'd8f9bf5d-0a2b-45e4-b98f-d186f401c73c';

-- residential-zoning | seg-354 @ 4338s | 22-word verbatim run | similarity 0.995
UPDATE essentials.quotes SET
    source_url  = 'https://on-the-record.onrender.com/meetings/f2cf80ef-a811-4d95-990d-b9c598284eb6?t=4338#seg-354',
    source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
  WHERE id = '84f56b9d-ca40-4268-b02a-81d4ac116ce0';


-- Guard: exactly 30 rows must now carry this source_name.
DO $$
DECLARE
    expected CONSTANT integer := 30;
    found integer;
BEGIN
    SELECT count(*) INTO found
      FROM essentials.quotes
     WHERE source_name = 'On the Record — 2026 LA Mayoral Debate (NBC LA)'
       AND source_url LIKE '%f2cf80ef-a811-4d95-990d-b9c598284eb6%'
       AND politician_id IN ('21c9e711-fb18-4afb-884f-08acd2b598ba', '26dbe16a-9dff-42c0-939f-5b5e529063ca')
       AND id IN (
               '2638f09b-b695-4c5b-a29c-a3c7cb739a10',
               'ce9bb5b9-ac51-4eec-8765-2ebe783e316d',
               'e294ef7e-5f85-42d6-9b89-cf018e817001',
               '8418a331-2eeb-418d-8473-fd36d5671276',
               'dd5bdce9-667c-4891-955a-98ec6c49d44e',
               '4c389a82-e480-4e6e-b4f4-fe8576ced385',
               'e0ec0298-540a-4ca7-be15-d8c07d4f1412',
               'e606a5f8-ff2f-4150-8d1b-60b81a4e8652',
               '0a1aacc9-3044-4b79-9e7a-a6095cc05824',
               '78ba1ed3-e7b3-471c-911b-4e4d49d87cde',
               '368cfab9-97e5-4ad6-afbf-477b4633e1cc',
               '132093b7-7acd-437f-a2fd-cee5af8f2704',
               '8edf4cb0-ed77-4e93-8ba0-a8dbdc04a09e',
               '71044f3e-380e-4887-9169-95fbd7926f08',
               '01c51a86-040c-414a-bdc4-c978d8b76c8e',
               'a6bb4672-eaf0-4f93-991c-040440c1394c',
               'f7625f7b-634f-49d4-9155-ab38455ba400',
               '644bc8e7-4545-4f50-b40e-0c2b6ae0a055',
               '9ea38e55-6e78-4ec1-b41c-eb74fa6a5217',
               '7d2e49cd-81bc-411b-bef6-498552588a1e',
               'ee3427dd-f91a-459f-8712-98117e3e428c',
               'c31d035f-deb2-4472-8b18-51e42d841244',
               'b9139e1d-824f-4670-b46c-caa624b2a83d',
               'baa16620-68a5-465e-85b9-71c9b4f5b6f0',
               'bdc80d45-40c2-4f1d-a0ef-e5619aab2d74',
               'eaaa2cdb-c56d-4be3-a798-b4e4e85c4836',
               '6c92a9fc-0082-4612-8f22-7d5f97097c09',
               'd7490082-efcf-4820-b571-8dc9a642f782',
               'd8f9bf5d-0a2b-45e4-b98f-d186f401c73c',
               '84f56b9d-ca40-4268-b02a-81d4ac116ce0'
           );

    IF found <> expected THEN
        RAISE EXCEPTION 'Aborting: expected % backfilled rows, found %.', expected, found;
    END IF;
END $$;

COMMIT;


-- ===========================================================================
-- SECTION 2 — essentials.discovered_sources (SEPARATE; NOT part of the
-- transaction above). Review and run independently, or skip entirely.
--
-- Registers the two NBCLA video assets for this debate. Neither currently has
-- a discovered_sources row (verified 2026-08-07). The first is the canonical
-- asset that meeting f2cf80ef was ingested from; the second is NBCLA's shorter
-- debate-only re-upload, registered so it is not later mistaken for a
-- separate event.
-- ===========================================================================

/*
INSERT INTO essentials.discovered_sources (
    source_key, url, title, channel_name, channel_id, channel_url,
    duration_seconds, published_at, matched_politician_ids, race_id,
    event_kind_guess, route, confidence, why, discovered_via, status
) VALUES
(
    'youtube:8rI3A6alVHM',
    'https://www.youtube.com/watch?v=8rI3A6alVHM',
    'LA Mayoral Debate (NBCLA) — Karen Bass, Spencer Pratt, Nithya Raman',
    'NBCLA',
    'UCSWoppsVL0TLxFQ2qP_DLqQ',
    'https://www.youtube.com/channel/UCSWoppsVL0TLxFQ2qP_DLqQ',
    6340,
    '2026-05-06T00:00:00Z',
    ARRAY['21c9e711-fb18-4afb-884f-08acd2b598ba', '26dbe16a-9dff-42c0-939f-5b5e529063ca']::uuid[],
    '9e888818-c50b-4c61-a106-a0839ff2479d',
    'debate',
    'ingest',
    1.0,
    'Canonical recording of the 2026-05-06 NBC4/Telemundo 52 Los Angeles mayoral debate, already ingested as meeting f2cf80ef-a811-4d95-990d-b9c598284eb6; all 30 previously unsourced Bass and Raman quotes were matched to it by exact contiguous word runs with speaker diarization confirming attribution in every case.',
    'agent',
    'pending'
),
(
    'youtube:-83WHHCKZDY',
    'https://www.youtube.com/watch?v=-83WHHCKZDY',
    'Full NBC4 broadcast: Karen Bass, Spencer Pratt, Nithya Raman debate for LA mayor',
    'NBCLA',
    'UCSWoppsVL0TLxFQ2qP_DLqQ',
    'https://www.youtube.com/channel/UCSWoppsVL0TLxFQ2qP_DLqQ',
    3474,
    '2026-05-07T00:00:00Z',
    ARRAY['21c9e711-fb18-4afb-884f-08acd2b598ba', '26dbe16a-9dff-42c0-939f-5b5e529063ca']::uuid[],
    '9e888818-c50b-4c61-a106-a0839ff2479d',
    'debate',
    'ingest',
    1.0,
    'NBCLA debate-only re-upload of the same 2026-05-06 mayoral debate as 8rI3A6alVHM; its auto-captions served as the independent second transcript confirming all 30 quotes, and it is registered here so the duplicate asset is not later ingested as a separate event.',
    'agent',
    'pending'
);
*/
