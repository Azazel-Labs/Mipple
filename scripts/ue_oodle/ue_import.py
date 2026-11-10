"""Editor Python (runs inside UnrealEditor-Cmd -run=pythonscript): import PNGs as Texture2D assets
configured for one Oodle encode each.

Driven by a JSON job file whose path is in env var UE_OODLE_JOB:
  {"images": ["C:/.../kodim18.png", ...], "formats": ["bc1","bc7",...], "lambdas": [0,1,10,...],
   "content_path": "/Game/Bench"}
Creates /Game/Bench/<fmt>/<name>_l<lambda> (name from job["names"], default the file stem) for every combination, saved to disk. The actual encode
happens when those packages are cooked (run_ue_oodle.py runs the cook commandlet afterwards).
Writes <job>.assets.json listing what was created and the settings read back from each asset.
"""
import json
import os
import unreal

job_path = os.environ["UE_OODLE_JOB"]
job = json.load(open(job_path))

TC = unreal.TextureCompressionSettings
COMPRESSION = {
    "bc1": TC.TC_DEFAULT,      # opaque source -> DXT1 (verify in cook log)
    "bc7": TC.TC_BC7,
    "bc5": TC.TC_NORMALMAP,    # BC5; UE may treat input as a normal map (see report)
    "bc4": TC.TC_ALPHA,        # single channel -> BC4 (verify in cook log)
}
LA = unreal.TextureLossyCompressionAmount
LCA = {0: LA.TLCA_NONE, 1: LA.TLCA_LOWEST, 10: LA.TLCA_LOW, 20: LA.TLCA_MEDIUM, 30: LA.TLCA_HIGH,
       40: LA.TLCA_HIGHEST}


def log(*a):
    unreal.log("UE_OODLE: " + " ".join(str(x) for x in a))


tools = unreal.AssetToolsHelpers.get_asset_tools()
created = []
for fmt in job["formats"]:
    for img in job["images"]:
        stem = job.get("names", {}).get(img) or os.path.splitext(os.path.basename(img))[0]
        for lam in job["lambdas"]:
            name = f"{stem}_l{lam}"
            dest = f"{job['content_path']}/{fmt}"
            t = unreal.AssetImportTask()
            t.filename = img.replace("\\", "/")
            t.destination_path = dest
            t.destination_name = name
            t.automated = True
            t.save = False
            t.replace_existing = True
            tools.import_asset_tasks([t])
            if not t.imported_object_paths:
                raise RuntimeError(f"import failed: {img}")
            tex = unreal.load_asset(t.imported_object_paths[0])
            props = {
                "mip_gen_settings": unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS,
                "srgb": False,
                "compression_settings": COMPRESSION[fmt],
                "lossy_compression_amount": LCA[lam],
                "never_stream": True,
                "virtual_texture_streaming": False,
                "power_of_two_mode": unreal.TexturePowerOfTwoSetting.NONE,
                "lod_group": unreal.TextureGroup.TEXTUREGROUP_WORLD,
            }
            if fmt == "bc5":
                props["flip_green_channel"] = False
            tex.set_editor_properties(props)
            # A fresh CompressionCacheId changes the texture's DDC key ("force the texture to be recompressed"),
            # so the cook really encodes (and dumps) even if identical source+settings were built before.
            tex.set_editor_property("compression_cache_id", unreal.GuidLibrary.new_guid())
            unreal.EditorAssetLibrary.save_loaded_asset(tex, False)
            back = {k: str(tex.get_editor_property(k)) for k in props}
            back["oodle_texture_sdk_version"] = str(tex.get_editor_property("oodle_texture_sdk_version"))
            back["compression_quality"] = str(tex.get_editor_property("compression_quality"))
            if fmt == "bc5":
                back["normalize_normals"] = str(tex.get_editor_property("normalize_normals"))
            back["source_size"] = [tex.blueprint_get_size_x(), tex.blueprint_get_size_y()]
            created.append({"fmt": fmt, "image": stem, "src": img, "lambda": lam,
                            "asset": f"{dest}/{name}", "settings": back})
            log("created", fmt, stem, lam, back)

json.dump(created, open(job_path + ".assets.json", "w"), indent=1)
log("done", len(created))
